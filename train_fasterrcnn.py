from fasterrcnn import FasterRCNN
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping 

def main(args):
    data_path = args.data_path
    devices = args.devices
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    patience = args.patience
    num_workers = args.workers
    opt = args.opt
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay
    norm_weight_decay = args.norm_weight_decay
    lr_scheduler = args.lr_scheduler
    lr_step_size = args.lr_step_size
    lr_steps = args.lr_steps
    lr_gamma = args.lr_gamma
    print_freq = args.print_freq
    output_dir = args.output_dir
    resume = args.resume
    start_epoch = args.start_epoch
    aspect_ratio_group_factor = args.aspect_ratio_group_factor
    trainable_backbone_layers = args.trainable_backbone_layers
    data_augmentation = args.data_augmentation
    sync_bn = args.sync_bn
    use_deterministic_algorithms = args.use_deterministic_algorithms
    weights = args.weights
    weights_backbone = args.weights_backbone
    amp = args.amp
    backend = args.backend
    use_v2 = args.use_v2


    kwargs = {"trainable_backbone_layers": trainable_backbone_layers}
    if data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True

    faster_rcnn = FasterRCNN(
        data_path=data_path,
        ann_folder="annotations/1classes",
        data_augmentation=data_augmentation,
        backend=backend,
        use_v2=use_v2,
        batch_size=batch_size,
        num_workers = num_workers,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        **kwargs
    )

    early_stop_callback = EarlyStopping(monitor="map@50", min_delta=0.00, patience=patience, verbose=False, mode="max")

    trainer = L.Trainer(
        devices=devices,
        max_epochs=max_epochs,
        enable_progress_bar=True,
        num_sanity_val_steps = 0,
        callbacks=[early_stop_callback]
    )

    trainer.fit(faster_rcnn)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="datasets/DFC2023/track1/", type=str, help="dataset path")

    parser.add_argument("--devices", default=1, type=int, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=2, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument("--max-epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--patience", default=10, type=int, metavar="N", help='number of checks with no improvement after which training will be stopped')
    parser.add_argument(
        "-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 4)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument(
        "--lr",
        default=0.02,
        type=float,
        help="initial learning rate, 0.02 is the default value for training on 8 gpus and 2 images_per_gpu",
    )
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--lr-scheduler", default="multisteplr", type=str, help="name of lr scheduler (default: multisteplr)"
    )
    parser.add_argument(
        "--lr-step-size", default=8, type=int, help="decrease lr every step-size epochs (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument("--print-freq", default=20, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=".", type=str, help="saved_models")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start_epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--aspect-ratio-group-factor", default=3, type=int)
    parser.add_argument("--rpn-score-thresh", default=None, type=float, help="rpn score threshold for faster-rcnn")
    parser.add_argument(
        "--trainable-backbone-layers", default=3, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )

    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--weights-backbone", default=None, type=str, help="the backbone weights enum name to load")

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # Use CopyPaste augmentation training parameter
    parser.add_argument(
        "--use-copypaste",
        action="store_true",
        help="Use CopyPaste data augmentation. Works only with data-augmentation='lsj'.",
    )

    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)