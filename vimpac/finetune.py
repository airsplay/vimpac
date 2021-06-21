import pickle
from collections import defaultdict, OrderedDict
import math
import os
import pprint
import subprocess
import time
import warnings

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

from vimpac.modeling import TransformerLayout
from vimpac.modeling_utils import PAD_TOKEN_ID
from vimpac.positional_embedding import option2args
from vimpac.param import get_arg_parser
from vimpac.data import get_dataset, DataTuple
from vimpac.metrics import TopkAccuracy
from utils import dist_all_gather_dict_avg, dict2markdown, MemorySafeModel, LabelSmoothingCrossEntropy


class Trainer:
    def __init__(self, args, num_labels):
        self.args = args
        self.gpu = self.args.gpu
        self.device = torch.device(f'cuda:{self.gpu}')
        self.visible = self.args.rank == 0                       # Only the first rank is visible.

        # Model
        pos_emb_args = option2args(args.pos_emb)
        self.classifier = torch.nn.Sequential(OrderedDict([
            ("backbone", TransformerLayout(
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
                pos_emb_args=pos_emb_args,
                layout=args.model[len("efflayout"):],
                args=args,
            )),
            ("dropout", nn.Dropout(self.args.last_dropout)),
            ("cls_fc", nn.Linear(self.args.hid_dim, num_labels)),
        ]))

        # Re-initialize the FC layer.
        torch.nn.init.constant_(self.classifier.cls_fc.weight.data, 0.0)
        torch.nn.init.constant_(self.classifier.cls_fc.bias.data, 0.0)

        if self.visible:
            print(self.classifier)

        # Load model
        if self.args.load is not None:
            self.load_model(self.args.load, from_pretrain=True, strict=False)
        if self.args.resume:
            # Resume option will override the load option!!
            self.load_model(os.path.join(self.args.output, "last", "classifier.pt"), from_pretrain=False, strict=True)

        # To Device
        self.classifier = self.classifier.to(self.device, non_blocking=True)
        self.classifier = torch.nn.parallel.DistributedDataParallel(
            self.classifier,
            device_ids=[self.gpu],
            broadcast_buffers=False,
            find_unused_parameters=False
        )

        # Cross entropy loss and accuracy meter.
        if self.args.label_smoothing is not None:
            self.criterion = LabelSmoothingCrossEntropy(self.args.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.top1 = TopkAccuracy(top_k=1)

        # Logging
        if self.visible:
            print("Criterion:", self.criterion)
            self.tb_writer = SummaryWriter(self.args.output)
            self.tb_writer.add_text("hyperparameters", dict2markdown(vars(self.args)))

        dist.barrier()

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        dset, loader, evaluator = train_tuple

        total_train_steps = len(loader) // self.args.gradient_accumulation * self.args.epochs
        # If warm_up < 1, it's a ratio; O.w., it's the steps.
        warm_up_steps = int(total_train_steps * self.args.warm_up) if self.args.warm_up < 1 else int(self.args.warm_up)

        no_decay = ["bias", "LayerNorm.weight", "ln_1.weight", "ln_2.weight", #"ln_3.weight", "ln_4.weight",
                    ".ln.weight", "ln_pre.weight", "ln_post.weight"]       # ln_1 / ln_2 are LayerNorm in GPT2
        self.optimized_params = [(name, param) for name, param in self.classifier.named_parameters()
                                 if param.requires_grad]

        if self.args.optim == "adamw":
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
        elif self.args.optim == "adam":
            optim = torch.optim.Adam(
                [p for n, p in self.optimized_params],
                lr=self.args.lr,
                betas=(self.args.beta1, self.args.beta2),
                eps=self.args.epsilon,
            )
        elif self.args.optim == "sgd":
            optim = torch.optim.SGD(
                [p for n, p in self.optimized_params],
                lr=self.args.lr,
                momentum=self.args.beta1,
                weight_decay=self.args.weight_decay,
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
            print(f"LR Scheduler: {self.args.lr_scheduler} LR scheduler with {total_train_steps} steps and"
                  f" {warm_up_steps} warm ups.")
            print(optim)

        # Resume the training
        if self.args.resume:
            starting_epoch = self.load_optim(self.args.output, "last", optim, lr_scheduler)
            if self.visible:
                print(f"Start to resume at epoch {starting_epoch}")
        else:
            starting_epoch = 0

        # FP16
        fp16_scaler = torch.cuda.amp.GradScaler(enabled=self.args.fp16)

        for epoch in range(starting_epoch, min(self.args.epochs, starting_epoch + self.args.interval)):
            tqdm_context = tqdm.tqdm if self.args.tqdm and self.visible else lambda x: x
            loader.sampler.set_epoch(epoch)
            train_result_dict = defaultdict(lambda: 0.)

            optim.zero_grad()
            for step, (imgs, labels) in enumerate(tqdm_context(loader)):

                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    codes = imgs.long()                    # O.w., the "imgs" are already the code
                    logits = self.classifier(codes)        # B, L, H, W --> B, num_video_dset_classes
                    loss = self.criterion(logits, labels)

                self.top1(logits, labels)
                fp16_scaler.scale(loss / self.args.gradient_accumulation).backward()

                if (step + 1) % self.args.gradient_accumulation == 0:
                    fp16_scaler.unscale_(optim)

                    train_result_dict["train/loss"] += loss.item()
                    grad_norm = torch.norm(
                        torch.stack([torch.norm(p.grad.detach(), 2) for p in self.classifier.parameters() if
                                     p.grad is not None]),
                        2).item()
                    if math.isnan(grad_norm):
                        train_result_dict["train/grad_nan_updates"] += 1
                    elif math.isinf(grad_norm):
                        train_result_dict["train/grad_inf_updates"] += 1
                    train_result_dict["train/gradnorm"] += grad_norm
                    train_result_dict["train/updates"] += 1
                    train_result_dict["train/pad_ratio"] += ((codes == PAD_TOKEN_ID).sum() / codes.numel()).item()
                    if self.args.clip_grad_norm > 1e-6:
                        torch.nn.utils.clip_grad_norm_(list(self.classifier.parameters()), self.args.clip_grad_norm)

                    fp16_scaler.step(optim)
                    fp16_scaler.update()

                    lr_scheduler.step()
                    optim.zero_grad()

                if self.args.debug:
                    break

            optim.zero_grad()

            if self.visible and (epoch + 1) % self.args.save_per_epochs == 0:  # Save pre-training results for every epochs.
                self.save_model(f"epoch{epoch + 1}", epoch=epoch, optimizer=optim, lr_scheduler=lr_scheduler)

            for key, value in train_result_dict.items():
                if "updates" not in key:
                    train_result_dict[key] = value / (len(loader) // self.args.gradient_accumulation)
            train_result_dict["train/accuracy"] = self.top1.get_metric(reset=True) * 100.

            result_dict = dict(train_result_dict)       # convert default_dict to dict for pickle
            result_dict = dist_all_gather_dict_avg(result_dict)

            # The exact eval will gather the results inside the evaluate protocal.
            if (epoch + 1) % self.args.eval_per_epochs == 0 or (epoch + 1) == self.args.epochs:   # Alway eval last epoch
                eval_result_dict = self.evaluate(eval_tuple)
                if self.visible:
                    result_dict.update(eval_result_dict)

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

                # Save the model
                self.save_model("last", epoch=epoch, optimizer=optim, lr_scheduler=lr_scheduler)

            dist.barrier()

    def evaluate(self, eval_tuple: DataTuple):
        self.classifier.eval()

        dset, loader, evaluator = eval_tuple

        sync_dict = defaultdict(list)
        tqdm_context = tqdm.tqdm if self.args.tqdm and self.visible else lambda x: x
        criterion = nn.CrossEntropyLoss(reduction="none").to(self.device)

        # Hack: close the lmdb files to avoid "Can not pickle LMDB Environment" issue.
        dset.close_lmdb_files()

        mem_safe_model = MemorySafeModel(self.classifier)

        for ids, imgs, labels in tqdm_context(loader):
            if len(imgs.shape) == 4:
                imgs = imgs.unsqueeze(1)       # if b, l, h, w; extend it to b, 1, l, h, w

            b, clips, l, h, w = imgs.shape

            imgs = imgs.to(self.device)         # Batch, clip, length, height, width
            imgs = torch.flatten(imgs, 0, 1)    # Batch x clip, length, height, width
            labels = labels.to(self.device)

            with torch.no_grad():
                codes = imgs.long()                     # O.w., the data from data loader would be the code
                logits = mem_safe_model(codes)          # Batch x clip, num_classes

                repeated_labels = torch.repeat_interleave(labels, clips)
                loss = criterion(logits, repeated_labels)       # (b x clips, N) , (b x clips, ) --> (b x clips,)
                loss = loss.reshape(b, clips).mean(-1)          # (b x clips,) --> (b, clips) --> (b, )

                # Calculate the accuracy with probs
                logits = logits.reshape(b, clips, -1)           # (b x clips, N) --> (b, clips, N)
                probs = F.softmax(logits, -1)                   # b, clips, softmax{num_classes}
                avg_probs = probs.mean(1)                       # b, clips, num_classes --> b, num_classes

                top_score, predictions = avg_probs.max(-1)

            sync_dict["ids"].append(ids)
            sync_dict["predictions"].append(predictions)
            sync_dict["loss"].append(loss)

            if self.args.debug:
                break

        # Gather the predictions from each batch, then gather over all nodes.
        for key, value in sync_dict.items():
            value = torch.cat(value, 0).cuda()
            tensor_list = [torch.empty_like(value) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list, value)
            sync_dict[key] = torch.cat(tensor_list)

        sync_dict_cpu = {key: value.cpu().numpy() for key, value in sync_dict.items()}

        # Gather the loss dict
        id2value_dict = {}
        for key, value in sync_dict_cpu.items():
            if key != "ids":
                id2value_dict[key] = {i: l for i, l in zip(sync_dict_cpu["ids"], value)}

        # Divide each loss component by number of val batches per GPU.
        if self.visible:
            # Assert different GPU's get the same prediction on overlapping data.
            for i, p in zip(sync_dict_cpu["ids"], sync_dict_cpu["predictions"]):
                assert id2value_dict["predictions"][i] == p

            results_dict = {
                "val/loss": sum(id2value_dict["loss"].values()) / len(id2value_dict["loss"]),
                "val/accuracy": evaluator.evaluate(id2value_dict["predictions"], force=not self.args.debug) * 100.,
            }
        else:
            results_dict = None

        self.classifier.train()
        dist.barrier()

        return results_dict

    def save_model(self, name, epoch=-1, optimizer: AdamW = None, lr_scheduler: LambdaLR = None):
        output_dir = os.path.join(self.args.output, name)
        os.makedirs(output_dir, exist_ok=True)

        # Save model configuration
        self.args.epoch = epoch
        pickle.dump(self.args, open(f"{output_dir}/args.pickle", 'wb'))
        with open(f"{output_dir}/args.txt", "w") as f:
            f.write(str(self.args))

        # Save model checkpoint
        model_to_save = (
            self.classifier.module if hasattr(self.classifier, "module") else self.classifier
        )  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "classifier.pt"))

        # Save Optimizer (and also the scheduler)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
        if lr_scheduler is not None:
            torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

    def load_model(self, model_path, from_pretrain=False, strict=False):
        if os.path.exists(model_path):
            if self.visible:
                print(f"Start to load model from {model_path}")
            model_dir = os.path.dirname(model_path)
            load_args = pickle.load(open(f"{model_dir}/args.pickle", 'rb'))
            assert load_args.pre_activation == self.args.pre_activation

            state_dict = torch.load(os.path.join(model_path), map_location=self.device)
            load_keys = set(state_dict.keys())

            # The vocab size might be different, we need to handle this here.
            # We always assume that the vocab_size are started from 0 to VOCAB_SIZE
            #   special tokens (e.g., [CLS], [MASK], [PAD]) are after VOCAB_SIZE.
            if "backbone.embedding.weight" in load_keys:
                load_vocab_size = state_dict["backbone.embedding.weight"].shape[0]
                current_vocab_size = self.classifier.backbone.embedding.weight.shape[0]
                # current_vocab_size = self.args.vocab_size
                if load_vocab_size != current_vocab_size:
                    assert load_vocab_size >= current_vocab_size
                    state_dict["backbone.embedding.weight"] = state_dict["backbone.embedding.weight"][:current_vocab_size]
                    if self.visible:
                        warnings.warn(f"We shrink the vocab size frm {load_vocab_size} to {current_vocab_size}."
                                      f"We assume that special tokens are after 0  ..  VOCAB_SIZE - 1 ({current_vocab_size - 1}). "
                                      f"E.g., [MASK] = VOCAB_SIZE, [PAD] = VOCAB_SIZE + 1")

            # If we need to change the shape of the positional embedding
            for key in load_keys:
                if key.startswith("backbone.positional_embedding"):
                    load_value = state_dict[key]       # (shape), dim
                    model_value = getattr(self.classifier.backbone.positional_embedding, key[len("backbone.positional_embedding."):])     # (model_shape), dim

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
                load_keys = set(state_dict.keys())
                model_keys = set(self.classifier.state_dict().keys())
                if load_keys != model_keys:
                    print("Weights in load but not in model")
                    for key in sorted(load_keys - model_keys):
                        print(f"\t{key}")

                    print("Weights in model but not in load")
                    for key in sorted(model_keys - load_keys):
                        print(f"\t{key}")

            self.classifier.load_state_dict(state_dict, strict=strict)
        else:
            if self.visible:
                print(f"{model_path} does not exist, do not load the weight.")
            if from_pretrain:
                raise FileNotFoundError(model_path)

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

    port = 9595
    while is_port_in_use(port):
        port += 1
    print("Use port", port)
    os.environ['MASTER_PORT'] = str(port)

    os.makedirs(args.output + "/results", exist_ok=True)
    if not args.load:
        pickle.dump(args, open(f"{args.output}/args.pickle", 'wb'))
        with open(f"{args.output}/args.txt", "w") as f:
            pprint.pprint(vars(args), f)

    pprint.pprint(vars(args))

    # Using all available gpus for multi-processing distributed
    args.gpus = torch.cuda.device_count()
    print("Use gpus ", list(range(args.gpus)))
    args.world_size = args.gpus    # * args.nodes
    args.rank = None                # Single-node distribution, the rank will be determined by the GPU id

    # Spawn
    mp.spawn(train, nprocs=args.gpus, args=(args,))


def slurm_multi_node_main(args):
    # local rank on the current node / global rank
    local_rank = int(os.environ['SLURM_LOCALID'])
    global_rank = int(os.environ['SLURM_PROCID'])

    # number of processes / GPUs per node
    world_size = int(os.environ['SLURM_NTASKS'])
    # n_gpu_per_node = world_size // n_nodes

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

    # Load the model architecture
    if args.load:
        model_dir = os.path.dirname(args.load)
        load_args = pickle.load(open(f"{model_dir}/args.pickle", 'rb'))

        def recover_and_print(args_name):
            load_value = getattr(load_args, args_name, None)
            now_value = getattr(args, args_name, None)
            if load_value != now_value:
                if args.rank == 0:
                    print(f"Reset the parameter {args_name} from {now_value} ---> {load_value}")
                setattr(args, args_name, load_value)

        model_keys = [
            "model",
            "pos_emb",
            "pre_activation",
            "layers",
            "hid_dim",
            "halve_mlp",
            "halve_att",
        ]
        if not args.different_shape:
            model_keys = model_keys + ["frame_size", "clip_len"]

        for args_name in model_keys:
            recover_and_print(args_name)

    # Set seed
    args.seed = args.seed + args.rank           # Make sure different nodes use different seed.
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
    bs_per_gpu = args.bs_per_gpu = args.batch_size // args.world_size // args.gradient_accumulation
    train_tuple, num_classes = get_dataset(
        args.dataset_name, "train", split_id=args.split_id,
        frame_size=args.frame_size, clip_len=args.clip_len, frame_rate=args.frame_rate,
        bs=bs_per_gpu, shuffle=True, drop_last=True, dist=True,
        num_workers=args.num_workers, is_train=True)
    valid_tuple, tmp_classes = get_dataset(
        args.dataset_name, "valid", split_id=args.split_id,
        frame_size=args.frame_size, clip_len=args.clip_len, frame_rate=args.frame_rate,
        num_test_clips=args.num_test_clips,
        bs=max(int(bs_per_gpu * 1.5 // args.num_test_clips), 1), shuffle=False, drop_last=False, dist=True,
        num_workers=args.num_workers, is_train=False)
    assert num_classes == tmp_classes


    if args.rank == 0:
        print(f"Load dataset {args.dataset_name}")

        print("Number in training", len(train_tuple.dataset))
        pprint.pprint(train_tuple.dataset.lmdb_paths)
        print("Number in validation", len(valid_tuple.dataset))
        pprint.pprint(valid_tuple.dataset.lmdb_paths)

        if args.frame_rate is not None:
            # assert train_tuple.dataset.sampled_frame_rate * args.frame_rate == 16
            actual_frame_rate = 16 / train_tuple.dataset.sampled_frame_rate
            print(f"Sample 1 frame per {train_tuple.dataset.sampled_frame_rate} frames from the original 16 FPS lmdb.")
            print(f"\t Thus the frame rate will be {actual_frame_rate}.")
            print(f"\t The duration is {args.clip_len / actual_frame_rate} seconds per clip.")
        print(f"Total batch size: {args.batch_size}")
        print(f"Gradient accumulation step: {args.gradient_accumulation}")
        print(f"Batch per gpu {bs_per_gpu}")
        print(f"Evaluation: bs per gpu {int(bs_per_gpu * 1.5 // args.num_test_clips)}, clip_nums {args.num_test_clips}")

    trainer = Trainer(args, num_classes)
    if args.eval_only:
        results = trainer.evaluate(valid_tuple)
        print(results)
    else:
        trainer.train(train_tuple, valid_tuple)
        if args.last_test_clips is not None:
            for num_test_clips in args.last_test_clips.split(','):
                if args.rank == 0:
                    print(f"Test with num test clips {num_test_clips}...")
                num_test_clips = int(num_test_clips)
                valid_tuple, tmp_classes = get_dataset(
                    args.dataset_name, "valid", split_id=args.split_id,
                    frame_size=args.frame_size, clip_len=args.clip_len, frame_rate=args.frame_rate,
                    num_test_clips=num_test_clips,
                    bs=max(int(bs_per_gpu * 1.5 // num_test_clips), 1), shuffle=False, drop_last=False, dist=True,
                    num_workers=args.num_workers, is_train=False)
                assert num_classes == tmp_classes
                results = trainer.evaluate(valid_tuple)
                if args.rank == 0:
                    print(f"Results of num test clips {num_test_clips}:")
                    print(results)

            time.sleep(20)


if __name__ == "__main__":
    # Parse args
    args = get_arg_parser().parse_args()

    if args.slurm_multinode_dist:
        slurm_multi_node_main(args)
    else:
        single_node_main(args)
