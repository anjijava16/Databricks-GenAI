import dataclasses
import hashlib
import logging
import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union

import pandas as pd

from databricks.rag_eval import context, env_vars, schemas, session
from databricks.rag_eval.datasets import entities as datasets_entities
from databricks.rag_eval.evaluation import entities as eval_entities
from databricks.rag_eval.utils import (
    error_utils,
    progress_bar_utils,
    rate_limit,
    spark_utils,
)

_logger = logging.getLogger(__name__)


_ANSWER_TYPES = [datasets_entities.SyntheticAnswerType.MINIMAL_FACTS]


@context.eval_context
def generate_evals_df(
    docs: Union[pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
    *,
    num_evals: int,
    agent_description: Optional[str] = None,
    question_guidelines: Optional[str] = None,
    guidelines: Optional[str] = None,  # Deprecated, but kept for backward compatibility
) -> pd.DataFrame:
    """
    Generate an evaluation dataset with synthetic requests and synthetic expected_facts, given a set of documents.

    The generated evaluation set can be used with `Databricks Agent Evaluation <https://docs.databricks.com/en/generative-ai/agent-evaluation/index.html>`_.

    For more details, see the `Synthesize evaluation set guide <https://docs.databricks.com/en/generative-ai/agent-evaluation/synthesize-evaluation-set.html>`_.

    Args:
        docs: A pandas/Spark DataFrame with a text column ``content`` and a ``doc_uri`` column.
        num_evals: The total number of evaluations to generate across all the documents. The function tries to distribute
            generated evals over all of your documents, taking into consideration their size. If num_evals is less than the
            number of documents, not all documents will be covered in the evaluation set.
        agent_description: Optional task description of the agent used to guide the generation.
        question_guidelines: Optional guidelines to guide the question generation. The string can be
            formatted in markdown and may include sections like:
            - User Personas: Types of users the agent should support
            - Example Questions: Sample questions to guide generation
            - Additional Guidelines: Extra rules or requirements

    """
    # Configs
    question_generation_rate_limit_config = rate_limit.RateLimitConfig(
        quota=env_vars.AGENT_EVAL_GENERATE_EVALS_QUESTION_GENERATION_RATE_LIMIT_QUOTA.get(),
        time_window_in_seconds=env_vars.AGENT_EVAL_GENERATE_EVALS_QUESTION_GENERATION_RATE_LIMIT_TIME_WINDOW_IN_SECONDS.get(),
    )
    answer_generation_rate_limit_config = rate_limit.RateLimitConfig(
        quota=env_vars.AGENT_EVAL_GENERATE_EVALS_ANSWER_GENERATION_RATE_LIMIT_QUOTA.get(),
        time_window_in_seconds=env_vars.AGENT_EVAL_GENERATE_EVALS_ANSWER_GENERATION_RATE_LIMIT_TIME_WINDOW_IN_SECONDS.get(),
    )
    max_workers = env_vars.RAG_EVAL_MAX_WORKERS.get()

    max_evals_per_chunk = env_vars.AGENT_EVAL_GENERATE_EVALS_MAX_NUM_EVALS_PER_CHUNK.get()
    max_tokens_per_chunk = env_vars.AGENT_EVAL_GENERATE_EVALS_MAX_NUM_TOKENS_PER_CHUNK.get()

    # Input validation
    if not isinstance(num_evals, int):
        raise error_utils.ValidationError("`num_evals` must be a positive integer.")
    if num_evals < 1:
        raise error_utils.ValidationError("`num_evals` must be at least 1.")
    if num_evals > env_vars.RAG_EVAL_MAX_INPUT_ROWS.get():
        raise error_utils.ValidationError(
            f"`num_evals` must be less than or equal to {env_vars.RAG_EVAL_MAX_INPUT_ROWS.get()}."
        )

    if agent_description is not None and not isinstance(agent_description, str):
        raise error_utils.ValidationError(
            f"Unsupported type for `agent_description`: {type(agent_description)}. "
            "`agent_description` must be a string."
        )
    if question_guidelines is not None and not isinstance(question_guidelines, str):
        raise error_utils.ValidationError(
            f"Unsupported type for `question_guidelines`: {type(question_guidelines)}. "
            "`question_guidelines` must be a string."
        )

    if guidelines is not None:
        _logger.warning(
            "`guidelines` is deprecated and will be removed in a future release. "
            "Please use `agent_description` and `question_guidelines` instead."
        )
    if guidelines is not None and not isinstance(guidelines, str):
        raise error_utils.ValidationError(
            f"Unsupported type for `guidelines`: {type(guidelines)}. " "`guidelines` must be a string."
        )
    if guidelines is not None and (agent_description or question_guidelines):
        raise error_utils.ValidationError(
            "Please remove `guidelines` and use `agent_description` and `question_guidelines` instead."
        )

    if guidelines is not None:
        question_guidelines = guidelines

    # Rate limiters
    question_generation_rate_limiter = rate_limit.RateLimiter.build_from_config(question_generation_rate_limit_config)
    answer_generation_rate_limiter = rate_limit.RateLimiter.build_from_config(answer_generation_rate_limit_config)

    generate_evals: List[eval_entities.EvalItem] = []
    docs: List[datasets_entities.Document] = _read_docs(docs)
    session.current_session().set_synthetic_generation_num_docs(len(docs))
    session.current_session().set_synthetic_generation_num_evals(num_evals)

    # Plan the generation tasks
    generation_tasks = _plan_generation_tasks(docs, num_evals, max_evals_per_chunk, max_tokens_per_chunk)

    # Use a progress manager to show the progress of the generation
    with progress_bar_utils.ThreadSafeProgressManager(
        total=num_evals,
        disable=False,
        desc="Generating evaluations",
        smoothing=0,  # 0 means using average speed for remaining time estimates
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} evals generated [Elapsed: {elapsed}, Remaining: {remaining}]",
    ) as progress_manager:
        with ThreadPoolExecutor(max_workers) as executor:
            futures = [
                executor.submit(
                    _generate_evals_for_doc,
                    task=task,
                    agent_description=agent_description,
                    question_guidelines=question_guidelines,
                    question_generation_rate_limiter=question_generation_rate_limiter,
                    answer_generation_rate_limiter=answer_generation_rate_limiter,
                    progress_manager=progress_manager,
                    current_session=session.current_session(),
                )
                for task in generation_tasks
            ]
            try:
                for future in as_completed(futures):
                    result = future.result()
                    generate_evals.extend(result)
            except KeyboardInterrupt:
                for future in futures:
                    future.cancel()
                _logger.info("Generation interrupted.")
                raise

    # Convert the evals to the new format with "inputs" and "expectations".
    rows: list[dict] = []
    for generate_eval in generate_evals:
        row = generate_eval.as_dict(use_chat_completion_request_format=True)
        row[schemas.INPUTS_COL] = row.pop(schemas.REQUEST_COL, {})
        row[schemas.EXPECTATIONS_COL] = {
            schemas.EXPECTED_FACTS_COL: row.pop(schemas.EXPECTED_FACTS_COL, []),
            schemas.EXPECTED_RETRIEVED_CONTEXT_COL: row.pop(schemas.EXPECTED_RETRIEVED_CONTEXT_COL, []),
        }
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def estimate_synthetic_num_evals(
    docs: Union[pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821,
    *,
    eval_per_x_tokens: int,
) -> int:
    """Estimate the number of evals to synthetically generate for full coverage over the documents.

    Args:
        docs: A pandas/Spark DataFrame with a text column ``content``.
        eval_per_x_tokens: Generate 1 eval for every x tokens to control the coverage level.
            500 tokens is ~1 page of text.

    Returns:
        The estimated number of evaluations to generate.
    """
    assert eval_per_x_tokens > 0, "eval_per_x_tokens must be positive."
    docs: List[datasets_entities.Document] = _read_docs(docs)
    total_num_tokens = sum(doc.num_tokens for doc in docs)
    num_evals = math.ceil(total_num_tokens / eval_per_x_tokens)
    return max(num_evals, len(docs))  # At least 1 eval per document


# ========================== Generation task planning ==========================
@dataclasses.dataclass
class _GenerationTask:
    doc: datasets_entities.Document
    """ The document to generate evaluations from. """
    num_evals_to_generate: int
    """ The number of evaluations to generate from this document. """


def _plan_generation_tasks(
    docs: List[datasets_entities.Document],
    num_evals: int,
    max_evals_per_chunk: int,
    max_tokens_per_chunk: int,
) -> List[_GenerationTask]:
    """
    Create an execution plan for synthetic generation.

    If the num_evals > num_docs, we distribute the number of evals to generate for each document based on the number of
    tokens in the document. The number of tokens in the document is used as a proxy for the amount of information
    in the document. Here is the high-level plan:
    - Sum up the tokens from all the docs and determine `tokens_per_eval = ceil(sum_all_tokens / num_evals)`
    - Walk each doc and generate `num_evals_for_doc = ceil(doc_tokens / tokens_per_eval)`
    - Stop early when we’ve generated `num_evals` in total

    If the num_evals <= num_docs, we randomly sample num_evals documents and generate 1 eval per document.

    :param docs: the list of documents to generate evaluations from
    :param num_evals: the number of evaluations to generate in total
    """
    if num_evals <= len(docs):
        return [_GenerationTask(doc=doc, num_evals_to_generate=1) for doc in random.sample(docs, num_evals)]
    else:
        sum_all_tokens_in_docs = sum(doc.num_tokens for doc in docs if doc.num_tokens)
        if sum_all_tokens_in_docs == 0:
            _logger.error("All documents have 0 tokens. No evaluations will be generated.")
            return []

        # This is potentially a float. However, we don't round here to reduce rounding error accumulation
        # Later, we'll round when deciding how many questions to allocate.
        tokens_per_eval = max(1, sum_all_tokens_in_docs / num_evals)

        generation_tasks: List[_GenerationTask] = []
        num_evals_planned = 0
        for doc in docs:
            if not doc.num_tokens:
                continue
            # Round up here; this errs in the right direction (generate more than num_evals, and then break early)
            # rather than accidentally generating too few evals.
            num_evals_for_doc = math.ceil(doc.num_tokens / tokens_per_eval)
            num_evals_for_doc = min(num_evals_for_doc, num_evals - num_evals_planned)  # Cap the number of evals
            generation_tasks.append(_GenerationTask(doc=doc, num_evals_to_generate=num_evals_for_doc))
            num_evals_planned += num_evals_for_doc
            # Stop early if we've generated enough evals
            if num_evals_planned >= num_evals:
                break
        generation_tasks = _maybe_break_up_tasks(
            generation_tasks,
            max_evals_per_chunk=max_evals_per_chunk,
            max_tokens_per_chunk=max_tokens_per_chunk,
        )
        return generation_tasks


def _maybe_break_up_tasks(
    tasks: List[_GenerationTask], max_evals_per_chunk: int, max_tokens_per_chunk
) -> List[_GenerationTask]:
    """Break up docs that have too many evals to generate.

    After breaking up tasks, the following should be true:
    - No task should use a doc larger than max_tokens_per_chunk tokens
    - No task should request more than max_evals_per_chunk evals

    Examples, assuming 5 max evals per chunk and 8k max tokens per chunk:
    - Request 20 evals from 8k tokens => 4x (Request 5 evals from 2k tokens)
    - Request 8 evals from 16k tokens => 2x (Request 4 evals from 8k tokens)
    - Request 20 evals from 16k tokens => 4x (Request 5 evals from 4k tokens)
    """
    new_tasks = []
    for task in tasks:
        if task.num_evals_to_generate <= max_evals_per_chunk and task.doc.num_tokens <= max_tokens_per_chunk:
            new_tasks.append(task)
        else:
            # Figure out which dimension we are more badly violating.
            eval_ratio = math.ceil(task.num_evals_to_generate / max_evals_per_chunk)
            token_ratio = math.ceil(task.doc.num_tokens / max_tokens_per_chunk)
            if eval_ratio > token_ratio:
                num_chunks = eval_ratio
                num_evals_per_chunk = max_evals_per_chunk
            else:
                num_chunks = token_ratio
                num_evals_per_chunk = math.ceil(task.num_evals_to_generate / num_chunks)

            # Floor, because consider the following case: doc_len = 100, num_chunks = 11.
            # Then, chars_per_chunk = 9.09; if we round up to 10, then we end up iterating over 10x10 chunks.
            # If our math was expecting to generate 11 chunks * 5 evals per chunk, then if we only generate 10 chunks,
            # we would end up with 50 evals instead of 55.
            # OTOH, if we round down to 9, then we end up with 11 chunks of 9 and a 12th chunk of length 1,
            # which ends up with 55 chunks as expected (the last chunk of size 1 never gets touched).
            chars_per_chunk = math.floor(len(task.doc.content) / num_chunks)
            chunked_docs = [
                datasets_entities.Document(
                    doc_uri=task.doc.doc_uri,
                    content=task.doc.content[offset : offset + chars_per_chunk],
                )
                for offset in range(0, len(task.doc.content), chars_per_chunk)
            ]
            num_evals_quota = task.num_evals_to_generate
            for doc in chunked_docs:
                if num_evals_quota <= 0:
                    break
                evals_to_generate = min(num_evals_per_chunk, num_evals_quota)
                new_tasks.append(
                    _GenerationTask(
                        doc=doc,
                        num_evals_to_generate=evals_to_generate,
                    )
                )
                num_evals_quota -= evals_to_generate
    return new_tasks


# ========================== Generation task execution ==========================
def _generate_evals_for_doc(
    task: _GenerationTask,
    agent_description: Optional[str],
    question_guidelines: Optional[str],
    question_generation_rate_limiter: rate_limit.RateLimiter,
    answer_generation_rate_limiter: rate_limit.RateLimiter,
    progress_manager: progress_bar_utils.ThreadSafeProgressManager,
    current_session: Optional[session.Session] = None,
) -> List[eval_entities.EvalItem]:
    """
    Generate evaluations for a single document.

    :param task: a generation task
    :param guidelines: optional guidelines to guide the question generation
    :param question_generation_rate_limiter: rate limiter for question generation
    :param answer_generation_rate_limiter: rate limiter for answer generation
    """
    session.set_session(current_session)

    doc = task.doc
    num_evals_to_generate = task.num_evals_to_generate

    if not doc.content or not doc.content.strip():
        _logger.warning(f"Skip {doc.doc_uri} because it has empty content.")
        return []

    # Get experiment ID from context
    experiment_id = context.get_context().get_mlflow_experiment_id()

    client = _get_managed_evals_client()
    with question_generation_rate_limiter:
        try:
            generated_questions = client.generate_questions(
                doc=doc,
                num_questions=num_evals_to_generate,
                agent_description=agent_description,
                question_guidelines=question_guidelines,
                experiment_id=experiment_id,
            )
        except Exception as e:
            _logger.warning(f"Failed to generate questions for doc {doc.doc_uri}: {e}")
            return []

    if not generated_questions:
        return []

    generated_answers: List[datasets_entities.SyntheticAnswer] = []
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(
                _generate_answer_for_question,
                question=question,
                answer_generation_rate_limiter=answer_generation_rate_limiter,
                current_session=current_session,
            )
            for question in generated_questions
        ]

        try:
            for future in as_completed(futures):
                result = future.result()
                generated_answers.append(result)
                progress_manager.update()
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            _logger.info("Generation interrupted.")
            raise
    return [
        eval_entities.EvalItem(
            question_id=hashlib.sha256(generated_answer.question.question.encode()).hexdigest(),
            question=generated_answer.question.question,
            raw_request=generated_answer.question.question,
            raw_response=None,
            ground_truth_answer=generated_answer.synthetic_ground_truth,
            ground_truth_retrieval_context=eval_entities.RetrievalContext(
                chunks=[
                    eval_entities.Chunk(
                        doc_uri=generated_answer.question.source_doc_uri,
                        content=generated_answer.question.source_context,
                    )
                ]
            ),
            expected_facts=generated_answer.synthetic_minimal_facts,
            source_id=generated_answer.question.source_doc_uri,
            source_type="SYNTHETIC_FROM_DOC",
        )
        for generated_answer in generated_answers
        if generated_answer is not None
    ]


def _generate_answer_for_question(
    question: datasets_entities.SyntheticQuestion,
    answer_generation_rate_limiter: rate_limit.RateLimiter,
    current_session: Optional[session.Session] = None,
) -> Optional[datasets_entities.SyntheticAnswer]:
    """
    Generate an answer for a single question.

    :param question: the question to generate an answer for
    :param answer_generation_rate_limiter: rate limiter for answer generation
    """
    session.set_session(current_session)
    if not question.question or not question.question.strip():
        # Skip empty questions
        return None

    # Get experiment ID from context
    experiment_id = context.get_context().get_mlflow_experiment_id()

    client = _get_managed_evals_client()
    with answer_generation_rate_limiter:
        try:
            return client.generate_answer(
                question=question,
                answer_types=_ANSWER_TYPES,
                experiment_id=experiment_id,
            )
        except Exception as e:
            _logger.warning(f"Failed to generate answer for question '{question.question}': {e}")
            return None


def _read_docs(
    docs: Union[pd.DataFrame, "pyspark.sql.DataFrame"],  # noqa: F821
) -> List[datasets_entities.Document]:
    """
    Read documents from the input pandas/Spark DateFrame.
    """
    if docs is None:
        raise error_utils.ValidationError("Input docs must not be None.")

    pd_df = spark_utils.normalize_spark_df(docs)

    if not isinstance(pd_df, pd.DataFrame):
        raise ValueError(
            f"Unsupported type for `docs`: {type(docs)}. "
            f"`docs` can be a pandas/Spark DataFrame with a text column `content` and a `doc_uri` column."
        )

    if "doc_uri" not in pd_df.columns or "content" not in pd_df.columns:
        raise error_utils.ValidationError("`docs` DataFrame must have 'doc_uri' and 'content' columns.")
    return [
        datasets_entities.Document(
            doc_uri=row["doc_uri"],
            content=row["content"],
        )
        for _, row in pd_df.iterrows()
    ]


# ================================ Misc. helpers ================================
def _get_managed_evals_client():
    """
    Get a managed evals client.
    """
    return context.get_context().build_managed_evals_client()
