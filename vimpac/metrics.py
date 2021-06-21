"""
Modified from https://github.com/kdexd/virtex, fix some bugs.
"""

import torch


class TopkAccuracy(object):
    r"""
    An accumulator for Top-K classification accuracy. This accumulates per-batch
    accuracy during training/validation, which can retrieved at the end. Assumes
    integer labels and predictions.
    .. note::
        If used in :class:`~torch.nn.parallel.DistributedDataParallel`, results
        need to be aggregated across GPU processes outside this class.
    Parameters
    ----------
    top_k: int, optional (default = 1)
        ``k`` for computing Top-K accuracy.
    """

    def __init__(self, top_k: int = 1):
        self._top_k = top_k
        self.reset()

    def reset(self):
        r"""Reset counters; to be used at the start of new epoch/validation."""
        self.num_total = 0.0
        self.num_correct = 0.0

    def __call__(self, predictions: torch.Tensor, ground_truth: torch.Tensor):
        r"""
        Update accumulated accuracy using the current batch.
        Parameters
        ----------
        ground_truth: torch.Tensor
            A tensor of shape ``(batch_size, )``, an integer label per example.
        predictions : torch.Tensor
            Predicted logits or log-probabilities of shape
            ``(batch_size, num_classes)``.
        """

        if self._top_k == 1:
            top_k = predictions.max(-1)[1].unsqueeze(-1)
        else:
            top_k = predictions.topk(min(self._top_k, predictions.shape[-1]), -1)[1]

        correct = top_k.eq(ground_truth.unsqueeze(-1)).float()

        self.num_total += correct[..., 0].numel()       # Support broadcasting between predictions and ground_truth
        self.num_correct += correct.sum()

    def get_metric(self, reset: bool = False):
        r"""Get accumulated accuracy so far (and optionally reset counters)."""
        if self.num_total > 1e-12:
            accuracy = float(self.num_correct) / float(self.num_total)
        else:
            accuracy = 0.0
        if reset:
            self.reset()
        return accuracy


