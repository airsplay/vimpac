# The code is modified from SimCLR (https://github.com/google-research/simclr) for NCE loss
#    and pytorch-lightning-bolts (https://github.com/PyTorchLightning/lightning-bolts/blob/5577453a6d7072724d9ae24184daf8f45d4baff7/pl_bolts/models/self_supervised/simclr/simclr_module.py#L20)
#      for the DistributedGather Operator.

import torch
import torch.distributed
import torch.nn.functional as F


class DifferentiableDistGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor: torch.Tensor):
        """
        Forward function will be the distributed.all_gather
        :param tensor: batch_size, hid_dim
        """
        ctx.input = tensor

        # [GPU1's (b, hid), GPU2's (b, hid), ...]
        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered_tensor, tensor)

        # [gpu_1_0, gpu_1_1, ..., gpu_2_0, gpu_2_1, ...]
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        backward function will be the distributed.reduce_scatter (with sum as reduce OP)
        :param grad_output: batch_size x world_size, hid_dim
        """
        grad_input = torch.zeros_like(ctx.input)
        per_gpu_batch_size = grad_input.shape[0]

        # Split the current grad into
        #   [(batch_size, hid_dim), ...]
        grad_output_list = list(grad_output.split(per_gpu_batch_size))

        # Reduce for each tensor and scatter them.
        torch.distributed.reduce_scatter(
            grad_input,
            grad_output_list,
            op=torch.distributed.ReduceOp.SUM,
            async_op=False)

        return grad_input


LARGE_NUM = 1e9


class DistributedNCELoss(torch.nn.Module):
    """
    Modified from the original TF implementation here:
    https://github.com/google-research/simclr/blob/master/objective.py
    """
    def forward(self,
                hidden,
                hidden_norm=True,
                temperature=1.0,
                dist=True):
        # Get (normalized) hidden1 and hidden2.
        if hidden_norm:
            hidden = hidden / hidden.norm(p=2, dim=-1, keepdim=True)
        hidden1, hidden2 = torch.chunk(hidden, chunks=2, dim=0)
        batch_size = hidden1.shape[0]

        # Gather hidden1/hidden2 across replicas and create local labels.
        if dist:
            hidden1_large = DifferentiableDistGather.apply(hidden1)
            hidden2_large = DifferentiableDistGather.apply(hidden2)
            enlarged_batch_size = hidden1_large.shape[0]
            replica_id = torch.distributed.get_rank()
            labels_idx = torch.arange(batch_size, device=hidden1.device) + replica_id * batch_size
            # labels = F.one_hot(labels_idx, enlarged_batch_size * 2)
            masks = F.one_hot(labels_idx, enlarged_batch_size).float()
        else:
            hidden1_large = hidden1
            hidden2_large = hidden2
            labels_idx = torch.arange(batch_size, device=hidden1.device)
            # labels = F.one_hot(labels_idx, batch_size * 2)
            masks = F.one_hot(torch.arange(batch_size, device=hidden1.device), batch_size).float()

        logits_aa = torch.matmul(hidden1, hidden1_large.t()) / temperature
        logits_aa = logits_aa - masks * LARGE_NUM       # Exclude the Xi . Xi
        logits_bb = torch.matmul(hidden2, hidden2_large.t()) / temperature
        logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, hidden2_large.t()) / temperature
        logits_ba = torch.matmul(hidden2, hidden1_large.t()) / temperature

        loss_func = torch.nn.CrossEntropyLoss()
        loss_a = loss_func(torch.cat([logits_ab, logits_aa], 1), labels_idx)
        loss_b = loss_func(torch.cat([logits_ba, logits_bb], 1), labels_idx)
        loss = (loss_a + loss_b) / 1.

        # Multiplied temperature to stablize the training?
        return loss * temperature


