from collections import defaultdict, OrderedDict
import math
import os
import pprint
import pickle
import subprocess
import time

import numpy as np
import torch
from torch import optim, nn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from transformers.optimization import LambdaLR

from vimpac.modeling import BertLMPredictionHead, ModelWithHead, CLSMLPHead, TransformerLayout
from vimpac.modeling_utils import PAD_TOKEN_ID
from vimpac.positional_embedding import option2args
from vimpac.masking import mask_3d, mask_3d_rand_replace
from vimpac.param import get_arg_parser
from vimpac.data import get_dataset, DataTuple
from vimpac.knn_monitor import KNNMonitor
from vimpac.nce_support import DistributedNCELoss
from utils import dist_all_gather_dict_avg, dict2markdown, interpolate_optimizer


class Trainer:
    def __init__(self, args):
        self.args = args
        self.gpu = self.args.gpu
        self.device = torch.device(f'cuda:{self.gpu}')
        self.visible = self.args.rank == 0                       # Only the first rank is visible.

        # Model
        pos_emb_args = option2args(args.pos_emb)
        transformer = TransformerLayout(
            height=args.frame_size // 8,
            width=args.frame_size // 8,
            length=args.clip_len,
            vocab_size=self.args.vocab_size + 5,
            hid_dim=self.args.hid_dim,
            layers=self.args.layers,
            heads=self.args.hid_dim // 64,
            dropout=self.args.dropout,
            use_cls_token=True,
            pre_activation=args.pre_activation,
            output_mode="both",  # Output the hidden of size [b, l, vocab_size]
            pos_emb_args=pos_emb_args,
            grad_checkpoint=self.args.grad_checkpoint,
            layout=args.model[len("efflayout"):],
            args=args,
        )
        if self.args.mlm_weight > 0.:
            mlm_head = BertLMPredictionHead(
                hid_dim=self.args.hid_dim,
                output_labels=self.args.vocab_size + 5,
                tied_embedding=transformer.embedding,
            )
        else:
            mlm_head = nn.Identity()
        if self.args.nce_weight > 0.:
            cls_head = CLSMLPHead(
                num_layers=self.args.nce_proj_layers,       # Default: 2 (as SimCLR)
                input_dim=self.args.hid_dim,
                hid_dim=self.args.nce_proj_hid_dim,         # Default: 4096
                output_dim=256,
                norm_type="bn",
                activation="relu",
                bn_after_proj=self.args.nce_bn_after_proj,  # Default: False
            )
        else:
            cls_head = nn.Identity()
        self.model = ModelWithHead(model=transformer, mlm_head=mlm_head, cls_head=cls_head)

        # Load model
        if self.args.load is not None:
            self.load_model(self.args.load, strict=True)
        if self.args.resume:
            # Resume option will override the load option!!
            self.load_model(os.path.join(self.args.output, "last", "classifier.pt"))

        # To Device
        self.model = self.model.to(self.device, non_blocking=True)
        self.model = torch.nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.gpu],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

        # Cross entropy loss and accuracy meter.
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.nce_criterion = DistributedNCELoss().to(self.device)

        # Logging
        if self.visible:
            print(self.model)
            self.tb_writer = SummaryWriter(self.args.output)
            self.tb_writer.add_text("hyperparameters", dict2markdown(vars(self.args)))

            def num_params(model):
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                return params

            print(f"Number of parameters in the whole model: {num_params(self.model)}")
            print(f"Number of parameters in the whole model (w/o embedding): "
                  f"{num_params(self.model) - num_params(self.model.module.backbone.embedding)}")
            print(f"Number of parameters in the backbone: {num_params(self.model.module.backbone)}")

        if self.args.knn_monitor:
            self.knn_monitor = KNNMonitor("ucf101", args, self.device, visible=False,
                                          tqdm=self.args.tqdm, dist=True)

        dist.barrier()

    def prepare_pretrain_input(self, codes: torch.Tensor):
        # Reduce the two-clip codes into one
        codes = codes.transpose(0, 1).flatten(0, 1)     # B, 2 --> 2, B --> 2B

        # Mask
        if not self.args.mask_rand_replace:
            masked_codes, labels = mask_3d(
                codes,
                mask_blocks=self.args.mask_blocks,
                pad_token_id=PAD_TOKEN_ID,
                mask_token_id=8196,
                vocab_size=8192,
            )  # 2 x (B, L, H, W)
        else:
            masked_codes, labels = mask_3d_rand_replace(
                codes,
                mask_blocks=self.args.mask_blocks,
                pad_token_id=PAD_TOKEN_ID,
                mask_token_id=8196,
                vocab_size=8192,
            )  # 2 x (B, L, H, W)

        return masked_codes, labels

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        dset, loader, evaluator = train_tuple

        total_train_steps = len(loader) // self.args.gradient_accumulation * self.args.epochs
        # If warm_up < 1, it's a ratio; O.w., it's the steps.
        warm_up_steps = int(total_train_steps * self.args.warm_up) if self.args.warm_up < 1 else int(self.args.warm_up)

        no_decay = [".bias",
                    "LayerNorm.weight",  ".ln.weight",
                    "ln_1.weight", "ln_2.weight", "ln_3.weight",
                    "ln_pre.weight", "ln_post.weight",
                    ]       # ln_1 / ln_2 are LayerNorm in GPT2
        self.optimized_params = [(name, param) for name, param in self.model.named_parameters()
                                 if param.requires_grad]

        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.optimized_params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.optimized_params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optim = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.lr,
            betas=(self.args.beta1, self.args.beta2),
            eps=self.args.epsilon,
        )

        if self.args.lr_scheduler == "linear":
            lr_scheduler = get_linear_schedule_with_warmup(
                optim,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_train_steps,
            )
        elif self.args.lr_scheduler == "cosine":
            lr_scheduler = get_cosine_schedule_with_warmup(
                optim,
                num_warmup_steps=warm_up_steps,
                num_training_steps=total_train_steps,
            )
        elif self.args.lr_scheduler == "constant":
            lr_scheduler = get_constant_schedule_with_warmup(
                optim,
                num_warmup_steps=warm_up_steps,
            )

        if self.visible:
            print()
            print(optim)
            print(f"LR Scheduler: {self.args.lr_scheduler} LR scheduler with {total_train_steps} steps and"
                  f" {warm_up_steps} warm ups; Peak learning rate at {self.args.lr}.")

        # Resume the training
        if self.args.resume:
            starting_epoch = self.load_optim(self.args.output, "last", optim, lr_scheduler)
            if self.visible:
                print(f"Start to train at epoch {starting_epoch}")
                print(f"Continue with learning rate {lr_scheduler.get_last_lr()}")
            # If the shape of the model changes, we interpolate the optimizer state as well.
            interpolate_optimizer(optim, self.visible)
        else:
            starting_epoch = 0

        # FP16
        fp16_scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        for epoch in range(starting_epoch, min(self.args.epochs, starting_epoch + self.args.interval)):
            tqdm_context = tqdm.tqdm if self.args.tqdm and self.visible else lambda x: x
            loader.sampler.set_epoch(epoch)
            train_result_dict = defaultdict(lambda: 0.)

            optim.zero_grad()
            for step, (codes, _) in enumerate(tqdm_context(loader)):

                codes = codes.long().to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    # labels are the original code for the masked codes.
                    # pos_ids are used to support sparse sampling, o.w., it will be None
                    masked_codes, labels = self.prepare_pretrain_input(codes)

                    # B, L, H, W --> B, L, H, W, vocab_size
                    cls_hiddens, logits = self.model(masked_codes)

                    loss = 0.
                    if self.args.mlm_weight > 0.:
                        # B x L x H x W, vocab_size (-2 is the W dimension) ; B x L x H x W
                        mlm_loss = self.args.mlm_weight * self.criterion(logits.flatten(0, -2), labels.flatten())
                        loss += mlm_loss
                    else:
                        mlm_loss = 0.
                    if self.args.nce_weight > 0.:
                        nce_loss = self.args.nce_weight * self.nce_criterion(
                            cls_hiddens,
                            hidden_norm=True,
                            temperature=0.2,
                            dist=True,
                        )
                        loss += nce_loss
                    else:
                        nce_loss = 0.

                fp16_scaler.scale(loss / self.args.gradient_accumulation).backward()

                # check parameters with no grad
                if self.visible:
                    for n, p in self.model.named_parameters():
                        if p.grad is None and p.requires_grad is True:
                            print("no grad", n, p.shape)  # prints unused parameters. Remove them from your model

                if (step + 1) % self.args.gradient_accumulation == 0:
                    fp16_scaler.unscale_(optim)

                    # Save to the logs
                    label_mask = (labels != -100).float()
                    train_result_dict["train/accu"] += (((logits.argmax(-1) == labels).float() * label_mask).sum()
                                                        / label_mask.sum()).item() * 100.
                    train_result_dict["train/loss"] += loss.item()
                    if self.args.mlm_weight > 0.:
                        train_result_dict["train/mlm_loss"] += mlm_loss.item()
                    if self.args.nce_weight > 0.:
                        train_result_dict["train/nce_loss"] += nce_loss.item()
                    train_result_dict["train/mask_ratio"] += (label_mask.sum() / label_mask.numel()).item()
                    grad_norm = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if
                                     p.grad is not None]),
                        2).item()
                    if math.isnan(grad_norm):
                        train_result_dict["train/grad_nan_updates"] += 1
                    elif math.isinf(grad_norm):
                        train_result_dict["train/grad_inf_updates"] += 1
                    else:
                        train_result_dict["train/gradnorm"] += grad_norm
                    train_result_dict["train/updates"] += 1
                    train_result_dict["train/pad_ratio"] += ((masked_codes == PAD_TOKEN_ID).sum() / masked_codes.numel()).item()

                    # Print Debug info
                    if epoch == 0 and train_result_dict["train/updates"] < 10 and self.visible:
                        codes = codes.transpose(0, 1).flatten(0, 1)
                        print()
                        print(f"Original codes shape: {codes.shape}")
                        b, l, h, w = codes.shape
                        flag = False
                        for l0 in range(1, l):
                            for h0 in range(1, h):
                                for w0 in range(1, w):
                                    if labels[0, l0, h0, w0] != -100:
                                        flag = True
                                        break
                                if flag:
                                    break
                            if flag:
                                break
                        check_range = 3
                        print("First masked location", l0, h0, w0, " at image with size", h, w)
                        for i in range(l0 - 1, min(l0 + 2, l)):
                            print(f"Tokens of time step {i}", codes.shape,
                                  codes[0, i, h0 - 1: h0 + check_range - 1, w0 - 1: w0 + check_range - 1])
                        for i in range(l0 - 1, min(l0 + 2, l)):
                            print(f"Masked Tokens of time step {i}", masked_codes.shape,
                                  masked_codes[0, i, h0 - 1: h0 + check_range - 1, w0 - 1: w0 + check_range - 1])
                        for i in range(l0 - 1, min(l0 + 2, l)):
                            print(f"Labels of time step {i}", labels.shape,
                                  labels[0, i, h0 - 1: h0 + check_range - 1, w0 - 1: w0 + check_range - 1])
                        print("Grad Norm", torch.norm(
                            torch.stack(
                                [torch.norm(p.grad.detach(), 2) for p in self.model.parameters() if
                                 p.grad is not None]),
                            2).item())
                        print("Flatten logits shape", logits.flatten(0, -2).shape)
                        print("Flatten labels shape", labels.flatten().shape)
                        print("Max and min token (original)", codes.max().item(), codes.min().item())
                        print("Max and min token (masked)", masked_codes.max().item(), masked_codes.min().item())
                        print("Max and min label", labels.max().item(), labels.min().item())
                        print("Masked Ratio:", ((labels != -100).sum() / labels.numel()).item())
                        print("Padding Ratio:", ((codes == PAD_TOKEN_ID).sum() / codes.numel()).item())

                        # NCE debug info
                        print(f"Loss: {loss}, MLM Loss: {mlm_loss}, NCE Loss: {nce_loss}")
                        print(f"cls_hiddens shape: {cls_hiddens.shape}")

                    if self.args.clip_grad_norm > 1e-6:
                        torch.nn.utils.clip_grad_norm_(list(self.model.parameters()), self.args.clip_grad_norm)

                    fp16_scaler.step(optim)
                    fp16_scaler.update()

                    lr_scheduler.step()
                    optim.zero_grad()

                    if self.args.debug and (step + 1) == self.args.gradient_accumulation * 1:
                        break

            optim.zero_grad()

            if self.visible:
                # Save the model before evaluation in case that the eval will be out of memory.
                self.save_model("last", epoch=epoch, optimizer=optim, lr_scheduler=lr_scheduler)
                if (epoch + 1) % self.args.save_per_epochs == 0:  # Save pre-training results for every epochs.
                    self.save_model(f"epoch{epoch + 1}", epoch=epoch, optimizer=optim, lr_scheduler=lr_scheduler)

            for key, value in train_result_dict.items():
                if "updates" not in key:
                    train_result_dict[key] = value / (len(loader) // self.args.gradient_accumulation)

            result_dict = dict(train_result_dict)       # convert default_dict to dict for pickle
            if (epoch + 1) % self.args.eval_per_epochs == 0:
                eval_result_dict = self.evaluate(eval_tuple)
                result_dict.update(eval_result_dict)
            result_dict = dist_all_gather_dict_avg(result_dict)

            if self.args.knn_monitor and (epoch + 1) % self.args.eval_per_epochs == 0 and not self.args.debug:
                self.model.eval()
                knn_top1, knn_top5 = self.knn_monitor.test(self.model.module.backbone, return_top5=True, output_mode="cls")
                result_dict["val/ucf101_knn_top1"] = knn_top1
                result_dict["val/ucf101_knn_top5"] = knn_top5
                self.model.train()

            if self.visible:
                log_str = f"Epoch: {epoch+1:03}, lr: {lr_scheduler.get_lr()}"
                for key in sorted(result_dict):
                    log_str += f" {key}: {result_dict[key]:0.4}"
                log_str += "\n"
                print(log_str)

                # Write the tensorboard
                for key, value in result_dict.items():
                    self.tb_writer.add_scalar(key, value, global_step=epoch + 1)     # TensorBoard
                self.tb_writer.flush()

            dist.barrier()
            torch.cuda.empty_cache()        # Empty cache before the next epoch

    def evaluate(self, eval_tuple: DataTuple):
        self.model.eval()

        dset, loader, evaluator = eval_tuple
        result_dict = defaultdict(lambda: 0.)

        tqdm_context = tqdm.tqdm if self.args.tqdm and self.visible else lambda x: x
        eval_num = 0
        try:
            for codes, _ in tqdm_context(loader):
                codes = codes.long().to(self.device)         # Batch, clip, length, height, width

                with torch.no_grad():
                    masked_codes, labels = self.prepare_pretrain_input(codes)

                    # B, L, H, W --> B, L, H, W, vocab_size
                    cls_hiddens, logits = self.model(masked_codes)

                    loss = 0.
                    if self.args.mlm_weight > 0.:
                        # B x L x H x W, vocab_size (-2 is the W dimension) ; B x L x H x W
                        mlm_loss = self.args.mlm_weight * self.criterion(logits.flatten(0, -2), labels.flatten())
                        loss += mlm_loss
                    if self.args.nce_weight > 0.:
                        nce_loss = self.args.nce_weight * self.nce_criterion(
                            cls_hiddens,
                            hidden_norm=True,
                            temperature=0.2,
                            dist=True,
                        )
                        loss += nce_loss

                    label_mask = (labels != -100).float()
                    result_dict["val/accu"] += (((logits.argmax(-1) == labels).float() * label_mask).sum().item()
                                                / max(label_mask.sum().item(), 1)) * 100.
                    result_dict["val/loss"] += loss.item()
                    if self.args.mlm_weight > 0.:
                        result_dict["val/mlm_loss"] += mlm_loss.item()
                    if self.args.nce_weight > 0.:
                        result_dict["val/nce_loss"] += nce_loss.item()
                    result_dict["val/mask_ratio"] += (label_mask.sum() / label_mask.numel()).item()
                    eval_num += 1

                if self.args.debug:
                    break
        except Exception as e:
            print(e)

        # Divide each loss component by number of val batches per GPU.
        for key, value in result_dict.items():
            result_dict[key] = value / max(eval_num, 1)

        self.model.train()

        return dict(result_dict)

    def save_model(self, name, epoch=-1, optimizer: AdamW = None, lr_scheduler: LambdaLR = None):
        output_dir = os.path.join(self.args.output, name)
        os.makedirs(output_dir, exist_ok=True)

        # Save model configuration
        self.args.epoch = epoch
        pickle.dump(self.args, open(f"{output_dir}/args.pickle", 'wb'))
        with open(f"{output_dir}/args.txt", "w") as f:
            pprint.pprint(vars(self.args), f)

        # Save model checkpoint
        model_to_save = (
            self.model.module if hasattr(self.model, "module") else self.model
        )  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "classifier.pt"))

        # Save Optimizer (and also the scheduler)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def load_model(self, model_path, from_pretrain=False, strict=False):
        if os.path.exists(model_path):
            state_dict = torch.load(os.path.join(model_path), map_location=self.device)
            load_keys = set(state_dict.keys())
            model_keys = set(self.model.state_dict().keys())

            # If we need to change the shape of the positional embedding
            for key in load_keys:
                if key.startswith("backbone.positional_embedding"):
                    load_value = state_dict[key]       # (shape), dim
                    model_value = getattr(self.model.backbone.positional_embedding, key[len("backbone.positional_embedding."):])     # (model_shape), dim

                    if load_value.shape != model_value.shape:
                        model_shape = model_value.shape[:-1]            #  (model_shape), dim --> (model_shape)

                        if self.visible:
                            print(f"Modifying key {key}")
                            print(f"\tshape before interpolation {load_value.shape}")

                        load_value = load_value.permute(-1, *range(len(model_shape))).unsqueeze(0)       # (shape), dim --> dim, (shape) --> 1, dim, (shape)
                        load_value = F.interpolate(load_value, model_shape, mode="linear", align_corners=False)
                        load_value = load_value.squeeze(0).permute(*[i+1 for i in range(len(model_shape))], 0)

                        if self.visible:
                            print(f"\tshape after interpolation {load_value.shape}")

                        state_dict[key] = load_value

            if self.visible:
                print(f"Start to load model from {model_path}")

                if load_keys != model_keys:
                    print("Weights in load but not in model")
                    for key in load_keys - model_keys:
                        print(f"\t{key}")

                    print("Weights in model but not in load")
                    for key in model_keys - load_keys:
                        print(f"\t{key}")
            self.model.load_state_dict(state_dict, strict=False)
        else:
            if strict:
                raise FileNotFoundError(model_path)
            else:
                if self.visible:
                    print(f"{model_path} does not exist, do not load the weight.")

    def load_optim(self, output_dir, name, optimizer: AdamW = None, lr_scheduler: LambdaLR = None):
        load_dir = os.path.join(output_dir, name)
        if not os.path.exists(load_dir):
            return 0

        if self.visible:
            print(f"Start to resume training from {load_dir}")

        # Save model configuration
        load_args = pickle.load(open(f"{load_dir}/args.pickle", 'rb'))
        assert load_args.optim == self.args.optim

        if optimizer is not None:
            optimizer.load_state_dict(
                torch.load(os.path.join(load_dir, "optimizer.pt"), map_location=self.device),
            )
            if self.visible:
                print(f"Load optimizer from {load_dir}/optimizer.pt")
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(
                torch.load(os.path.join(load_dir, "scheduler.pt"), map_location=self.device),
            )
            if self.visible:
                print(f"Load scheduler from {load_dir}/scheduler.pt")

        return load_args.epoch + 1          # Will start from the next epoch


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


def single_node_main(args):
    os.environ['MASTER_ADDR'] = '127.0.0.1'

    os.makedirs(args.output + "/results", exist_ok=True)
    pickle.dump(args, open(f"{args.output}/args.pickle", 'wb'))
    with open(f"{args.output}/args.txt", "w") as f:
        pprint.pprint(vars(args), f)

    pprint.pprint(vars(args))

    port = 9595
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    # Using all available gpus for multi-processing distributed
    args.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(args.gpus)))
    args.world_size = args.gpus
    args.rank = None                # Single-node distribution, the rank will be determined by the GPU id

    # Spawn
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def slurm_multi_node_main(args):
    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])

    # define master address and master port
    hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
    master_addr = hostnames.split()[0].decode('utf-8')

    # set environment variables for 'env://'
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(29500)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)

    args.world_size = world_size
    args.rank = global_rank

    if args.rank == 0:
        os.makedirs(args.output + "/results", exist_ok=True)
        pickle.dump(args, open(f"{args.output}/args.pickle", 'wb'))
        with open(f"{args.output}/args.txt", "w") as f:
            pprint.pprint(vars(args), f)

        pprint.pprint(vars(args))

    print(f"Launch global rank {global_rank}, local rank {local_rank}, world size {world_size}; "
          f"Master addr {master_addr}, master port {29500}")

    train(local_rank, args)


def train(gpu, args):
    args.gpu = gpu
    if args.rank is None:
        args.rank = args.gpu        # If single-node distribution, set rank to the gpu_id
    torch.cuda.set_device(gpu)

    # Set seed
    args.seed = args.seed + args.rank           # Make sure different servers use different seed.
    torch.manual_seed(args.seed)
    import random
    import numpy
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    # Init process group in each process
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank
    )

    # Data
    if args.bs_per_gpu is not None:
        args.gradient_accumulation = max(args.batch_size // args.world_size // args.bs_per_gpu, 1)
    assert args.batch_size % (args.world_size * args.gradient_accumulation) == 0
    bs_per_gpu = args.batch_size // args.world_size // args.gradient_accumulation
    train_tuple, _ = get_dataset(
        args.dataset_name, "train",
        frame_size=args.frame_size, clip_len=args.clip_len,
        num_train_clips=2,
        relevance_range=args.nce_relevance_range,
        bs=bs_per_gpu, shuffle=True, drop_last=True, dist=True,
        num_workers=args.num_workers, is_train=True, only_one_aug=True)

    # If pretrained with HT100M, use UCF101 val as the validation
    val_dataset = "ucf101" if args.dataset_name == "howto100m" else args.dataset_name

    # For pre-training validation, we
    #   1. distributed eval
    #   2. do not return multiple crop/clips as in fine-tuning
    #   3. Only use the canonical "center/frame-size/nohflip" crop.
    valid_tuple, _ = get_dataset(
        val_dataset, "valid",
        frame_size=args.frame_size, clip_len=args.clip_len,
        num_train_clips=2,
        relevance_range=args.nce_relevance_range,
        bs=int(bs_per_gpu * 1.5), shuffle=False, drop_last=False, dist=True,
        num_workers=args.num_workers, is_train=True, only_one_aug=True)

    if args.rank == 0:
        print(f"Load dataset {args.dataset_name}")
        print("Number in training", len(train_tuple.dataset))
        pprint.pprint(train_tuple.dataset.lmdb_paths)
        print("Number in validation", len(valid_tuple.dataset))
        pprint.pprint(valid_tuple.dataset.lmdb_paths)
        print(f"Total batch size: {args.batch_size}")
        print(f"Gradient accumulation step: {args.gradient_accumulation}")
        print(f"Batch size per step: {bs_per_gpu * args.world_size}")
        print(f"World size {args.world_size}")
        print(f"Batch per gpu {bs_per_gpu}")
        print(f"Test batch per gpu {int(bs_per_gpu * 2)}")

    trainer = Trainer(args)
    trainer.train(train_tuple, valid_tuple)

    time.sleep(10)


if __name__ == "__main__":
    # Parse args
    args = get_arg_parser().parse_args()

    if args.slurm_multinode_dist:
        slurm_multi_node_main(args)
    else:
        single_node_main(args)
