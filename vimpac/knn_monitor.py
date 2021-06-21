"""
KNN monitor for test tracking the current feature quality.

Copied from
https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
Note:
    This moco demo add F.normalize instead of directly using the features as the below two code bases.

The code is further from https://github.com/leftthomas/SimCLR and https://github.com/zhirongw/lemniscate.pytorch.
"""

import pickle
import torch
import torch.nn.functional as F
import torch.distributed as dist
import tqdm

from vimpac.data import get_dataset


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    if type(data) is torch.Tensor:
        data = data.cpu()
    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data = pickle.loads(buffer)
        if type(data) is torch.Tensor:
            data = data.to("cuda")
        data_list.append(data)

    return data_list


class KNNMonitor:
    def __init__(self, dataset: str, args, device: torch.device, visible: bool = False, tqdm: bool = False, dist=False):
        """

        :param dataset: the dataset name of this monitor.
        """
        bs_per_gpu = args.batch_size // args.world_size // args.gradient_accumulation
        train_tuple, num_classes = get_dataset(
            dataset, "train",
            frame_size=args.frame_size, clip_len=args.clip_len,
            bs=bs_per_gpu, shuffle=False, drop_last=False, dist=dist,
            num_workers=args.num_workers, is_train=True, only_one_aug=True)
        self.memory_data_loader = train_tuple.loader
        valid_tuple, tmp_classes = get_dataset(
            dataset, "valid",
            frame_size=args.frame_size, clip_len=args.clip_len,
            num_test_clips=args.num_test_clips,
            bs=bs_per_gpu, shuffle=False, drop_last=False, dist=dist,
            num_workers=args.num_workers, is_train=True, only_one_aug=True)
        self.test_data_loader = valid_tuple.loader

        assert num_classes == tmp_classes

        self.num_classes = num_classes
        # This temperature param was tuned for Imagnet, from https://github.com/zhirongw/lemniscate.pytorch
        self.knn_t = 0.07

        self.visible = visible
        self.tqdm = tqdm
        self.dist = dist
        self.device = device
        self.knn_k = 200

    def test(self, net, return_top5=False, **model_args):
        classes = self.num_classes
        total_top1, total_top5, total_num, feature_bank, target_bank = 0.0, 0.0, 0, [], []
        with torch.no_grad():
            # generate feature bank
            for data, target in tqdm.tqdm(self.memory_data_loader, desc='Feature extracting', disable=not self.visible or not self.tqdm):
                feature = net(data.long().cuda(non_blocking=True), **model_args)
                feature = F.normalize(feature, dim=1)
                feature_bank.append(feature)
                target_bank.append(target)
            # [D, N]
            feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
            target_bank = torch.cat(target_bank, dim=0).contiguous()

            if self.dist:
                feature_banks = all_gather(feature_bank)
                target_banks = all_gather(target_bank)
                first_device = feature_banks[0].device
                for bank in feature_banks:
                    if first_device != bank.device:
                        raise Warning(f"Device are not equal {first_device} and {bank.device}.")
                # [[feature_dim, batch_size]] --> [feature_dim, sum(batch_size)]
                feature_bank = torch.cat(feature_banks, -1)
                target_bank = torch.cat(target_banks, 0)
                if self.visible:
                    print(f"Gather feature bank for device {self.device}: {feature_bank.shape}")

            # [N]
            # feature_labels = torch.tensor(self.memory_data_loader.dataset.targets, device=feature_bank.device)
            feature_labels = target_bank

            # if self.dist:
            #     feature_labels = feature_labels.reshape(-1, dist.get_world_size()).t().flatten().contiguous()

            # loop test data to predict the label by weighted knn search
            test_bar = tqdm.tqdm(self.test_data_loader, disable=not self.visible or not self.tqdm)
            for data, target in test_bar:
                data, target = data.long().cuda(non_blocking=True), target.cuda(non_blocking=True)
                feature = net(data, **model_args)
                feature = F.normalize(feature, dim=1)

                pred_labels = self.knn_predict(feature, feature_bank, feature_labels, classes)

                total_num += data.size(0)
                total_top1 += (pred_labels[:, 0] == target).float().sum().item()
                total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

        if self.dist:
            total_top1 = sum(all_gather(total_top1))
            total_top5 = sum(all_gather(total_top5))
            total_num = sum(all_gather(total_num))

        if return_top5:
            return total_top1 / total_num * 100, total_top5 / total_num * 100
        else:
            return total_top1 / total_num * 100

    # knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
    # implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
    def knn_predict(self, feature, feature_bank, feature_labels, classes):
        # compute cos similarity between each feature vector and feature bank ---> [B, N]
        #   Note: [batch_size, feature_dim] x [feature_dim, memory_bank_size] --> [batch_size, memory_bank_size]
        sim_matrix = torch.mm(feature, feature_bank)
        # [B, K]
        sim_weight, sim_indices = sim_matrix.topk(k=self.knn_k, dim=-1)
        # [B, K]
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
        sim_weight = (sim_weight / self.knn_t).exp()

        # counts for each class
        one_hot_label = torch.zeros(feature.size(0) * self.knn_k, classes, device=sim_labels.device)
        # [B*K, C]
        one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
        # weighted score ---> [B, C]
        pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

        pred_labels = pred_scores.argsort(dim=-1, descending=True)
        return pred_labels

