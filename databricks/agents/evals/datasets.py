""" "
WARNING: `databricks.agents.evals.datasets` package is deprecated. Use `databricks.agents.datasets`
instead.
"""

import logging

from databricks.rag_eval.datasets.managed_evals import (
    add_evals,
    add_tags,
    create_evals_table,
    delete_evals,
    delete_evals_table,
    get_eval_labeling_config,
    get_ui_links,
    grant_access,
    list_tags,
    revoke_access,
    sync_evals_to_uc,
    tag_evals,
    untag_evals,
    update_eval_labeling_config,
)

logging.warning(
    "DeprecationWarning: `databricks.agents.evals.datasets` package is deprecated. "
    "Use `databricks.agents.datasets` instead."
)

__all__ = [
    "create_evals_table",
    "delete_evals_table",
    "update_eval_labeling_config",
    "get_eval_labeling_config",
    "add_evals",
    "delete_evals",
    "list_tags",
    "add_tags",
    "get_ui_links",
    "grant_access",
    "sync_evals_to_uc",
    "revoke_access",
    "tag_evals",
    "untag_evals",
]
