import pickle

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

import torch.nn.functional as F


def dist_all_gather_dict_avg(dict_obj, world_size=None, rank=None):
    """
    Code copied from https://github.com/pytorch/pytorch/issues/3473#issuecomment-627361795
    :param dict_obj: a python dict to be synced up
    :param world_size: will use the default value if none
    :param rank: Will use the default value if None
    :return:
    """
    if world_size is None:
        world_size = dist.get_world_size()
    if rank is None:
        rank = dist.get_rank()

    MAX_LENGTH = 8 * 1024       # 8K
    tensor_list = [torch.empty(MAX_LENGTH, dtype=torch.uint8).cuda() for _ in range(world_size)]

    data = pickle.dumps(dict_obj)

    # Encode
    data_length = len(data)
    data = data_length.to_bytes(4, "big") + data
    assert len(data) < MAX_LENGTH
    data += bytes(MAX_LENGTH - len(data))
    data = np.frombuffer(data, dtype=np.uint8)
    assert len(data) == MAX_LENGTH
    tensor = torch.from_numpy(data).cuda()
    assert len(tensor) == MAX_LENGTH

    # Distributed Gather, see https://pytorch.org/tutorials/intermediate/dist_tuto.html
    # for details.
    dist.all_gather(tensor_list, tensor)

    # Decode
    results = []
    for tensor in tensor_list:
        data = tensor.cpu().numpy().tobytes()
        length = int.from_bytes(data[:4], "big")
        data = data[4:length + 4]
        results.append(pickle.loads(data))

    assert len(results) == world_size

    reduced_result = results[0]
    for result in results[1:]:
        for key in reduced_result:
            reduced_result[key] += result[key]
    reduced_result = {key: value / world_size for key, value in reduced_result.items()}

    return reduced_result


def dict2markdown(key2value):
    header = "| key | value |"
    separator = "| --- | --- |"

    content = [f"| {key} | {key2value[key]} |" for key in sorted(key2value)]

    lines = [header,
             separator,
             *content]

    return "\n".join(lines)


class MemorySafeModel(torch.nn.Module):
    def __init__(self, model):
        super(MemorySafeModel, self).__init__()
        self.model = model
        self.max_bs = 9595

    def forward(self, *args, **kwargs):
        """

        :param args: args are tensors.
        :param kwargs: kwargs are controlling arguments (e.g., True/False)
        :return:
        """
        num_data = len(args[0])
        while True:
            try:
                outputs = []
                for start_idx in range(0, num_data, self.max_bs):
                    end_idx = start_idx + self.max_bs
                    small_args = [arg[start_idx: end_idx] for arg in args]
                    output = self.model(*small_args, **kwargs)
                    outputs.append(output)
                if type(outputs[0]) is tuple:
                    outputs = zip(*outputs)
                    outputs = [torch.cat(output, 0) for output in outputs]
                else:
                    outputs = torch.cat(outputs, 0)
                break
            except RuntimeError as e:
                if self.max_bs == 1:
                    raise RuntimeError("The GPU can not support even 1 batch size!")
                else:
                    previous_bs = min(self.max_bs, num_data)
                    self.max_bs = max(previous_bs // 2, 1)
                    print(f"Downgrade the batch size from {previous_bs} to {self.max_bs}")
        return outputs


def interpolate_optimizer(optimizer, visible=False):
    for group in optimizer.param_groups:
        for p in group["params"]:
            if p not in optimizer.state:
                continue
            param = p.data
            state = optimizer.state[p]

            model_shape = param.shape[:-1]  # (model_shape), dim --> (model_shape)
            for key in ["exp_avg", "exp_avg_sq"]:

                load_value = state[key]
                if load_value.shape != param.shape:
                    if visible:
                        print(f"Modifying param with shape {param.shape} of key {key}")
                        print(f"\tshape before interpolation {load_value.shape}")

                    load_value = load_value.permute(-1, *range(len(model_shape))).unsqueeze(
                        0)  # (shape), dim --> dim, (shape) --> 1, dim, (shape)
                    load_value = F.interpolate(load_value, model_shape, mode="linear", align_corners=False)
                    load_value = load_value.squeeze(0).permute(*[i + 1 for i in range(len(model_shape))], 0)

                    if visible:
                        print(f"\tshape after interpolation {load_value.shape}")

                    state[key] = load_value



class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

    def extra_repr(self) -> str:
        return f"label smoothing: {self.smoothing}"


if __name__ == "__main__":
    results = {"accu": 0.2,
               "path": "sdafsdfadf"}
    print(dict2markdown(results))
