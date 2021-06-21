# Test the correctness of multi-node NCE loss.
# We compute the grad of single-node NCE and multi-node NCE and compare their grads.
import os

import numpy as np
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from vimpac.nce_support import DistributedNCELoss


def single_node_main():
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    port = 9595
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    gpus = torch.cuda.device_count()

    np.random.seed(95)
    batch_size = 64
    dim = 128
    tensor = np.random.random((batch_size, dim))

    # Spawn
    mp.spawn(train, nprocs=gpus, args=(gpus, tensor,))


def train(gpu, gpus:int, tensor: torch.Tensor):
    torch.cuda.set_device(gpu)

    # Set seed
    # Init process group in each process
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=gpus,
        rank=gpu,
    )


    x = torch.from_numpy(tensor).cuda()
    x.requires_grad = True

    x_ = x.clone().detach()
    x_.requires_grad = True

    bs_per_gpu = x.shape[0] // gpus
    slice_obj = slice(bs_per_gpu * gpu, bs_per_gpu * (gpu + 1))
    print(slice_obj)

    loss = DistributedNCELoss()(
        x,
        hidden_norm=True,
        temperature=1.0,
        dist=False)
    loss.backward()
    x_local_grad = torch.cat((
        x.grad[bs_per_gpu // 2 * gpu: bs_per_gpu // 2 * (gpu + 1)],
        x.grad[x.shape[0] // 2 + bs_per_gpu // 2 * gpu: x.shape[0] // 2 + bs_per_gpu // 2 * (gpu + 1)],
    ))
    if gpu == 0:
        print(x_local_grad.shape)
        print(x_local_grad)

    x_ = torch.cat((
        x_[bs_per_gpu // 2 * gpu: bs_per_gpu // 2 * (gpu + 1)],
        x_[x.shape[0] // 2 + bs_per_gpu // 2 * gpu: x.shape[0] // 2 + bs_per_gpu // 2 * (gpu + 1)],
    )).clone().detach()
    x_.requires_grad = True
    loss = DistributedNCELoss()(
        x_,
        hidden_norm=True,
        temperature=1.0,
        dist=True)
    loss.backward()
    if gpu == 0:
        print(x_.grad.shape)
        print(x_.grad)
        print(x_local_grad / x_.grad)
        print((x_local_grad / x_.grad).max())
        print((x_local_grad / x_.grad).min())


if __name__ == "__main__":
    single_node_main()


