from typing import Any
import torch
import torchvision

from dataset import get_dataset
import presets
import utils

from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer, seed_everything

from coco_eval import CocoEvaluator
from dataset import get_coco_api_from_dataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def get_transform(is_train, data_augmentation="hflip", backend="pil", use_v2=False):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=data_augmentation, backend=backend, use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)


data_path = "datasets/DFC2023/track1"
data_augmentation="hflip"
backend="pil"
use_v2=False
batch_size = 2
num_workers = 4
trainable_backbone_layers = 3
rpn_score_thresh = None

kwargs = {"trainable_backbone_layers": trainable_backbone_layers}
if data_augmentation in ["multiscale", "lsj"]:
    kwargs["_skip_resize"] = True

if rpn_score_thresh is not None:
    kwargs["rpn_score_thresh"] = rpn_score_thresh

dataset_train, num_classes, num_domains = get_dataset(data_path, 
                                        transforms=get_transform(True, data_augmentation=data_augmentation, backend=backend, use_v2=use_v2), 
                                        ann_folder="annotations/1classes", image_set='train')

dataset_val, _, _ = get_dataset(data_path, 
                                        transforms=get_transform(False, backend=backend, use_v2=use_v2), 
                                        ann_folder="annotations/1classes", image_set='val')

train_sampler = torch.utils.data.RandomSampler(dataset_train)
val_sampler = torch.utils.data.SequentialSampler(dataset_val)

train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, batch_size, drop_last=True)

train_collate_fn = utils.collate_fn

data_loader_train = torch.utils.data.DataLoader(
    dataset_train, batch_sampler=train_batch_sampler, num_workers=num_workers, collate_fn=train_collate_fn
)

data_loader_val = torch.utils.data.DataLoader(
    dataset_val, batch_size=1, sampler=val_sampler, num_workers=num_workers, collate_fn=utils.collate_fn
)

def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


seed_everything(25081992)

class FasterRCNN(LightningModule):
    
    def __init__(self, num_classes):
        super().__init__()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes+1, **kwargs)
        coco = get_coco_api_from_dataset(data_loader_val.dataset)
        iou_types = _get_iou_types(self.detector)
        self.coco_evaluator = CocoEvaluator(coco, iou_types)
        self.lr = 0.02


    def configure_optimizers(self):
      optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
      return optimizer
    
    def training_step(self, batch, batch_idx):
        images, targets = batch

        images = list(image for image in images)
        targets = [{k: v if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = self.detector(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        loss = sum(loss for loss in loss_dict.values())
        return {"loss": loss, "log": loss_dict}
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(img for img in images)
        outputs = self.detector(images)
        outputs = [{k: v for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

    def on_validation_start(self):
        pass

    def on_validation_epoch_end(self):
        print("on validation epoch end: ", self.current_epoch)
        if self.current_epoch > 0:
            map50 = self.coco_evaluator.coco_eval['bbox'].stats[1]
            print("map50", map50)


early_stop_callback = EarlyStopping(
   monitor='val_accuracy',
   min_delta=0.00,
   patience=3,
   verbose=False,
   mode='max'
)

detector = FasterRCNN(1)

trainer = Trainer(devices=1, enable_progress_bar=True, max_epochs=2,deterministic=False)

trainer.fit(detector,data_loader_train, data_loader_val)