import os
from dg_maskrcnn import DGMaskRCNN
from lightning.pytorch import Trainer

def main(args):

    mask_rcnn = DGMaskRCNN.load_from_checkpoint(checkpoint_path=os.path.join(args.checkpoint_dir, args.checkpoint_file_name+'.ckpt'))

    trainer = Trainer()

    trainer.test(mask_rcnn)

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=add_help)
    
    parser.add_argument("--checkpoint-dir", default="checkpoints", type=str, help="checkpoint directory")
    parser.add_argument("--checkpoint-file-name", default="best_prop", type=str, help="the name of checkpoint file")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)