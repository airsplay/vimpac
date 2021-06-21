import argparse


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='train VAE for DALLE-pytorch')

    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--eval-only', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--tqdm', action="store_true")
    parser.add_argument('--output', type=str, default=".", help='experiment output path')
    parser.add_argument("--seed", type=int, default=9595)
    parser.add_argument("--split-id", type=int, default=1, help="train_val split id for specifying HMDB51 and UCF101")

    # Data
    parser.add_argument('--dataset-name', type=str, default="ucf101", choices=[
        "kinetics400",
        "ucf101",
        "hmdb51",
        "ssv2",
        "howto100m",
        "diving48",
    ])
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')

    # Training
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs (default: 500)')
    parser.add_argument('--interval', type=int, default=9595, help="How many epochs trained in this experiment")
    parser.add_argument("--eval-per-epochs", type=int, default=1)
    parser.add_argument("--save-per-epochs", type=int, default=9595)    # a large number thus will not save!
    parser.add_argument('--fp16', action="store_true")
    parser.add_argument('--optim', default='adamw', choices=["adamw", "sgd", "adam"])
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-scheduler', type=str, default="linear", choices=["linear", "constant", "cosine"])
    parser.add_argument('--warm-up', type=float, default=0.05,
                        help="warm up ratio; If < 1, it's a ratio; O.w., it's the actual steps.")
    parser.add_argument('--beta1', type=float, default=0.9, help='adam beta 1')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam beta 2')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='adam epsilon')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='adam epsilon')
    parser.add_argument('--clip-grad-norm', type=float, default=0.)
    parser.add_argument('--gradient-accumulation', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--bs-per-gpu', type=int, default=None, help='batch size per gpu')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size for each update')
    parser.add_argument('--label-smoothing', type=float, default=None)
    parser.add_argument('--grad-checkpoint', action="store_true")

    # Input
    parser.add_argument('--frame-size', type=int, default=128, help='image size for training (default: 256)')
    parser.add_argument('--clip-len', type=int, default=10, help='number of frames in each clip')
    parser.add_argument('--frame-rate', type=int, default=None, help='The frame rate of sampled clip.')
    parser.add_argument("--different-shape", action="store_true",
                        help="Allow using different shape from pre-training in fine-tuning.")
    parser.add_argument("--num-test-clips", type=int, default=5, help="the number of clips used in test")
    parser.add_argument("--last-test-clips", type=str, default=None,
                        help="the number of clips tested at the end of training. Frames are split by comma. E.g., "
                             "2,5,10 means that frames 2, 5, 10 will be tested in the end.")
    parser.add_argument('--vocab-size', type=int, default=2048, help='number of code')

    # Transformer Model
    parser.add_argument("--model", type=str, default="bert", choices=[
        "efflayoutT,H,W",
        "efflayoutT",
        "efflayoutT,H",
        "efflayoutT,H|W",
        "efflayoutT|H|W",
        "efflayoutH|W,T",
    ])
    parser.add_argument("--pos-emb", type=str, default="default", choices=[
        "default",
        "temporal_separation",
        "hw_separation",
        "pre_defined",
        "frozen",
    ])
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hid-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--last-dropout", type=float, default=0.0, help="")
    parser.add_argument("--pre-activation", action="store_true", help="Trigger pre-activation; Ablation study only.")
    parser.add_argument("--halve-mlp", action="store_true", help="Halve the MLP intermediate dimensions")
    parser.add_argument("--halve-att", action="store_true", help="Halve the attention head dimensions")

    # Only for Pre-training
    parser.add_argument("--mask-blocks", default=-1, type=int,
                        help="The number of 3D blocks randomly masked. -1 means token-level uniform masking.")
    parser.add_argument("--mask-rand-replace", action="store_true",
                        help="If True, use the BERT-style masking: 80% mask, 10% keep, 10% rand token.")
    parser.add_argument("--knn-monitor", action="store_true", help="KNN Monitor in UCF101.")
    parser.add_argument("--slurm-multinode-dist", action="store_true",
                        help="It will trigger the slurm's multi-node distributed training.")

    # NCE Pre-training
    parser.add_argument("--mlm-weight", type=float, default=1., help="The weight of masked language modeling loss")
    parser.add_argument("--nce-weight", type=float, default=1., help="The weight of contrastive loss")
    parser.add_argument("--nce-proj-layers", type=int, default=2, help="Number of layers in NCE head")
    parser.add_argument("--nce-proj-hid-dim", type=int, default=4096)
    parser.add_argument("--nce-bn-after-proj", action="store_true",
                        help="Use an additional BN after the last proj layer")
    parser.add_argument("--nce-relevance-range", type=int, default=None,
                        help="The upper bound of distance between two positive clips")

    return parser
