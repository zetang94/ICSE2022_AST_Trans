from ignite.exceptions import NotComputableError
from ignite.metrics import Metric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
import numpy as np
import json
from utils import EOS_WORD
from valid_metrices.compute_scores import eval_accuracies
from valid_metrices.google_bleu import compute_bleu


__all__ = ['BLEU4', 'bleu_output_transform', 'TotalMetric']


def bleu_output_transform(outputs, nl_i2w):
    y_pred, y = outputs
    batch_size = y.size(0)
    references = []
    hypothesises = []
    for i in range(batch_size):
        reference = [nl_i2w[c.item()] for c in y[i]]
        if EOS_WORD in reference:
            reference = reference[:reference.index(EOS_WORD)]
        hypothesis = [nl_i2w[c.item()] for c in y_pred[i]]
        if EOS_WORD in hypothesis:
            hypothesis = hypothesis[:hypothesis.index(EOS_WORD)]
        if len(hypothesis) == 0:
            hypothesis = ['<???>']
        if len(reference) == 0:
            continue
        references.append(reference)
        hypothesises.append(hypothesis)

    return hypothesises, references


class TotalMetric(Metric):
    def __init__(self, output_path, output_transform=lambda x: x, device="cpu"):
        super(TotalMetric, self).__init__(output_transform=output_transform, device=device)
        self._hypothesises = None
        self._references = None
        self.output_path = output_path

    @reinit__is_reduced
    def reset(self):
        self._hypothesises = []
        self._references = []

    @reinit__is_reduced
    def update(self, output):
        hypothesises, references = output
        self._hypothesises.extend(hypothesises)
        self._references.extend(references)

    @sync_all_reduce("hypothesises", "references")
    def compute(self):
        if len(self._hypothesises) == 0:
            raise NotComputableError("Rouge must have "
                                     "at least one example before it can be computed.")
        hypothesises = {index: [' '.join(value)] for index, value in enumerate(self._hypothesises)}
        references = {index: [' '.join(value)] for index, value in enumerate(self._references)}

        bleu, rougle_l, meteor, ind_bleu, ind_rouge = eval_accuracies(hypothesises, references)

        outputs = []
        for i in hypothesises.keys():
            outputs.append({
                'predict': hypothesises[i][0],
                'true': references[i][0],
                'bleu': ind_bleu[i],
                'rouge': ind_rouge[i]
            })
        with open(self.output_path + 'predict_results.json', 'w') as f:
            json.dump(outputs, f)

        return bleu, rougle_l, meteor


class BLEU4(Metric):
    def __init__(self, output_transform=lambda x: x, device="cpu"):
        super(BLEU4, self).__init__(output_transform=output_transform, device=device)
        self._bleu_scores = None
        self._num_examples = None

    @reinit__is_reduced
    def reset(self):
        self._bleu_scores = 0
        self._num_examples = 0

    @staticmethod
    def batch_bleu(predicts, trues):
        scores = []
        for i in range(len(trues)):
            bleu_score = compute_bleu([[trues[i]]], [predicts[i]], smooth=True)[0]
            scores.append(bleu_score)
        return scores

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        scores = self.batch_bleu(y_pred, y)
        self._bleu_scores += np.sum(scores)
        self._num_examples += len(scores)

    @sync_all_reduce("_bleu_scores", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError("BLEU4 must have "
                                     "at least one example before it can be computed.")
        return self._bleu_scores / self._num_examples

