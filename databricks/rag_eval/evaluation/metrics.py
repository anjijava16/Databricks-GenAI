import abc
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import mlflow
import pandas as pd

from databricks.rag_eval import constants, schemas, session
from databricks.rag_eval.evaluation import entities

_logger = logging.getLogger(__name__)


class Metric(abc.ABC):
    """
    Metric represents a method to compute a metric of an evaluation.
    """

    @abc.abstractmethod
    def run(
        self,
        *,
        eval_item: Optional[entities.EvalItem] = None,
        assessment_results: Optional[List[entities.AssessmentResult]] = None,
    ) -> List[entities.MetricResult]:
        """
        Run the metric on a single eval item and produce a list of metric results.
        A single eval item can produce multiple metric results since multiple metrics can be batch computed
        together for a single EvalItem.

        :param eval_item: The eval item to assess.
        :param assessment_results: The assessment results for the eval item.
        :return: A list of metric results.
        """
        pass


def compute_eval_metrics(
    *,
    eval_item: entities.EvalItem,
    assessment_results: List[entities.AssessmentResult],
    metrics: List[Metric],
) -> List[entities.MetricResult]:
    """
    Compute the per-eval-item metrics.
    """
    if not metrics:
        return []

    metric_results: List[entities.MetricResult] = []
    parent_session = session.current_session()

    def run_metric(metric):
        session.set_session(parent_session)
        return metric.run(eval_item=eval_item, assessment_results=assessment_results)

    # Use a thread pool to run metrics in parallel
    # Use the number of metrics as the number of workers
    with ThreadPoolExecutor(max_workers=len(metrics)) as executor:
        futures = [executor.submit(run_metric, metric) for metric in metrics]

        try:
            for future in as_completed(futures):
                result = future.result()
                metric_results.extend(result)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print("Metrics computation interrupted.")
            raise
    return metric_results


# ================================ Built-in Metrics ================================


class LatencyMetric(Metric):
    """
    Compute the latency (in fractional seconds to a microsecond granularity) from the trace information.
    """

    def run(
        self,
        *,
        eval_item: Optional[entities.EvalItem] = None,
        assessment_results: Optional[List[entities.AssessmentResult]] = None,
    ) -> List[entities.MetricResult]:
        if not eval_item:
            return []
        if eval_item.trace is None or eval_item.trace.info is None or eval_item.trace.info.execution_duration is None:
            return []
        return [
            entities.MetricResult.make_legacy_metric(
                metric_name=schemas.LATENCY_SECONDS_COL,
                metric_value=eval_item.trace.info.execution_duration / 1000.0,
            )
        ]


class GroundTruthRetrievalMetric(Metric):
    """
    Compute the ground truth retrieval metrics for an eval item.

    The ground truth retrieval metrics include: precision, recall, etc.

    The metrics is calculated based on the doc_uri of retrieval context and ground truth retrieval context
    in the eval item.

    This metric outputs the following:
    - The recall for the whole context (K = length of retrieval)
    """

    def run(
        self,
        *,
        eval_item: Optional[entities.EvalItem] = None,
        assessment_results: Optional[List[entities.AssessmentResult]] = None,
    ) -> List[entities.MetricResult]:
        results = []
        if not eval_item or not eval_item.retrieval_context or not eval_item.ground_truth_retrieval_context:
            return results

        retrieved_docs = eval_item.retrieval_context.get_doc_uris()
        ground_truth_docs = eval_item.ground_truth_retrieval_context.get_doc_uris()
        if not retrieved_docs or not ground_truth_docs:
            return results

        k = len(retrieved_docs)
        for metric_name in constants.GROUND_TRUTH_RETRIEVAL_METRIC_NAMES:
            mlflow_eval_metric = getattr(mlflow.metrics, f"{metric_name}_at_k")(k)

            eval_fn = mlflow_eval_metric.eval_fn
            try:
                metric_value = eval_fn(pd.Series([retrieved_docs]), pd.Series([ground_truth_docs]))
                score = metric_value.scores[0]
                results.append(
                    entities.MetricResult.make_legacy_metric(
                        metric_name=f"{schemas.GROUND_TRUTH_DOCUMENT_PREFIX}{metric_name}",
                        metric_value=score,
                    )
                )
            except Exception as e:
                full_metric_name = schemas.GROUND_TRUTH_DOCUMENT_PREFIX + metric_name
                _logger.debug(f"Error in computing {full_metric_name} for eval_item {eval_item}: {e}")

        return results


class LlmJudgedRetrievalMetric(Metric):
    """
    Compute the LLM-judged precision metrics using the results of the retrieval assessment.

    We use the positional_rating of the retrieval assessment results to compute the precision at k metrics.
    """

    def run(
        self,
        *,
        eval_item: Optional[entities.EvalItem] = None,
        assessment_results: Optional[List[entities.AssessmentResult]] = None,
    ) -> List[entities.MetricResult]:
        results = []
        for assessment_result in assessment_results or []:
            if not isinstance(assessment_result, entities.PerChunkAssessmentResult):
                continue
            ratings = [
                rating
                for _, rating in assessment_result.positional_rating.items()
                if rating.categorical_value is not None
            ]
            if not ratings:
                continue
            precision = sum(r.categorical_value == entities.CategoricalRating.YES for r in ratings) / len(ratings)
            results.append(
                entities.MetricResult.make_legacy_metric(
                    metric_name=schemas.get_retrieval_llm_metric_col_name(
                        f"{assessment_result.assessment_name}/precision"
                    ),
                    metric_value=precision,
                )
            )
        return results


LATENCY_METRIC = LatencyMetric()
GROUND_TRUTH_RETRIEVAL_METRIC = GroundTruthRetrievalMetric()
LLM_JUDGED_RETRIEVAL_METRIC = LlmJudgedRetrievalMetric()

BUILT_IN_METRICS = [
    LATENCY_METRIC,
    GROUND_TRUTH_RETRIEVAL_METRIC,
    LLM_JUDGED_RETRIEVAL_METRIC,
]
