from dg_maskrcnn import DGMaskRCNN
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch import Trainer, seed_everything
from dgfrcnn import DGFasterRCNN

def main(args):

    kwargs = {"trainable_backbone_layers": args.trainable_backbone_layers}
    if args.data_augmentation in ["multiscale", "lsj"]:
        kwargs["_skip_resize"] = True

    faster_rcnn = DGFasterRCNN(
        data_path=args.data_path,
        ann_folder=args.ann_folder,
        data_augmentation=args.data_augmentation,
        backend=args.backend,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        opt=args.opt,
        lr_steps=args.lr_steps,
        lr_gamma=args.lr_gamma,
        image_dg=args.image_dg,
        ins_dg=args.ins_dg,
        **kwargs
    )

    seed_everything(25081992)

    early_stop_callback= EarlyStopping(monitor='map@50', min_delta=0.00, patience=args.patience, verbose=False, mode='max')
    checkpoint_callback = ModelCheckpoint(monitor='map@50', dirpath=args.checkpoint_dir, filename=args.checkpoint_file_name, mode='max')

    trainer = Trainer(max_epochs=args.max_epochs, callbacks=[early_stop_callback, checkpoint_callback], num_sanity_val_steps=0)

    trainer.fit(faster_rcnn)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)

    parser.add_argument("--data-path", default="datasets/DFC2023/track1/", type=str, help="dataset path")

    parser.add_argument("--ann-folder", default="annotations/1classes", type=str, help="the folder name of the annotation files")

    parser.add_argument("--checkpoint-dir", default="checkpoints", type=str, help="checkpoint directory")

    parser.add_argument("--checkpoint-file-name", default="best_prop", type=str, help="the name of checkpoint file")

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
        "--lr-steps",
        default=[16, 22],
        nargs="+",
        type=int,
        help="decrease lr every step-size epochs (multisteplr scheduler only)",
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma (multisteplr scheduler only)"
    )
    parser.add_argument(
        "--trainable-backbone-layers", default=3, type=int, help="number of trainable layers of backbone"
    )
    parser.add_argument(
        "--data-augmentation", default="hflip", type=str, help="data augmentation policy (default: hflip)"
    )
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")

    parser.add_argument('--image-dg', action="store_true", help="whether the image level domain generalization is included during training")
    parser.add_argument('--ins-dg', action="store_true", help="whether the instance level domain generalization is included during training")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)