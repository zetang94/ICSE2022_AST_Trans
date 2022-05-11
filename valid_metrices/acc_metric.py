import torch
from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce


__all__ = ['MatchAccMetric']


class MatchAccMetric(Metric):
    def __init__(self, pad, need_mask, output_transform=lambda x: x, device=None):
        super(MatchAccMetric, self).__init__(output_transform, device)
        self.pad = pad
        self.need_mask = need_mask
        self._match_token = None
        self._total_token = None

    @reinit__is_reduced
    def reset(self):
        self._match_token = 0
        self._total_token = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        if self.need_mask:
            y_pred.masked_fill_(y == self.pad, self.pad)

        pad_num = torch.sum(y == self.pad)
        total_num = torch.sum(y != self.pad)
        equal_num = torch.sum(y_pred == y)

        self._match_token += equal_num - pad_num
        self._total_token += total_num

    @sync_all_reduce('_match_token', '_total_token')
    def compute(self):
        if self._total_token == 0:
            raise NotComputableError('MatchAccMetric must have at least one example before it can be computed.')
        return self._match_token / self._total_token

