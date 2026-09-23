from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class IterOffsetDisplayHook(Hook):
    """Display cumulative iteration number with an offset.

    This hook prints an extra log line like "IterDisplay: [12100/160000]" at a
    fixed interval, without touching the runner's real counters or the default
    TextLoggerHook. Use it when you load weights via load_from (not resuming
    optimizer) and still want the log to reflect the cumulative progress.

    Args:
        iter_offset (int): The starting offset to add to current iter.
        total (int | None): The total number to display. If None, displays
            (runner.max_iters + iter_offset).
        interval (int): Logging interval (iterations).
    """

    def __init__(self, iter_offset=0, total=None, interval=100):
        super().__init__()
        self.iter_offset = int(iter_offset)
        self.total = None if total is None else int(total)
        self.interval = int(interval)

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        cur = (runner.iter + 1) + self.iter_offset
        total = self.total if self.total is not None else (
            runner.max_iters + self.iter_offset)
        # Print an additional concise line.
        runner.logger.info(f'IterDisplay: [{cur}/{total}]')













