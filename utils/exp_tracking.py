from ignite.contrib.engines import common
import ignite.distributed as idist
from clearml import Task


has_clearml = True


@idist.one_rank_only()
def _clearml_log_params(params_dict):
    task = Task.current_task()
    task.connect(params_dict)


if has_clearml:
    log_params = _clearml_log_params
    setup_logging = common.setup_clearml_logging
else:
    raise RuntimeError(
        "No experiment tracking system is setup. "
        "Please, setup either ClearML. "
    )
