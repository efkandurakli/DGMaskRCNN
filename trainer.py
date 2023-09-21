import torch
import torchvision
import lightning as L
from datamodule import RoofDetectionDataModule
from coco_eval import CocoEvaluator
from dataset import get_coco_api_from_dataset
from dataset import get_dataset
import presets
from torch.utils.data import DataLoader, Subset
import utils

def get_transform(is_train, data_augmentation="hflip", backend="pil", use_v2=False):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=data_augmentation, backend=backend, use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)

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

class FasterRCNN(L.LightningModule):
    def __init__(
            self, 
            data_path = "datasets/DFC2023/track1",
            ann_folder = "annotations/1classes",
            data_augmentation="hflip",
            backend="pil",
            use_v2=False,
            batch_size = 2,
            num_workers = 4,
            lr: float = 0.01,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            **kwargs,
        ):
        super().__init__()

        self.train_dataset, self.num_classes, self.num_domains = get_dataset(data_path, 
                                transforms=get_transform(True, data_augmentation=data_augmentation,
                                                         backend=backend, use_v2=use_v2), 
                                ann_folder=ann_folder, image_set="train")
        
        
        self.val_dataset, _, _ = get_dataset(data_path, 
                                transforms=get_transform(False, data_augmentation=data_augmentation,
                                                         backend=backend, use_v2=use_v2), 
                                ann_folder=ann_folder, image_set="val")
        

        self.save_hyperparameters()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=self.num_classes+1, trainable_backbone_layers=3, **kwargs)

        self.automatic_optimization = False

    def forward(self, images):
        return self.detector(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = self.detector(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        self.log("train_loss", losses_reduced)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(losses)
        opt.step()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(img for img in images)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        outputs = self(images)
        outputs = [{k: v for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)


    def test_step(self, batch, batch_idx):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_validation_epoch_start(self):
        coco = get_coco_api_from_dataset(self.val_dataset)
        iou_types = _get_iou_types(self.detector)
        self.coco_evaluator = CocoEvaluator(coco, iou_types)

    def on_validation_end(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

    def on_test_start(self):
        pass

    def on_test_end(self):
        pass

    def configure_optimizers(self):
        parameters = [p for p in self.detector.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        return optimizer
    
    def train_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, sampler=train_sampler, 
                          num_workers=self.hparams.num_workers, collate_fn=utils.collate_fn)

    def val_dataloader(self):
        val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=1, sampler=val_sampler, 
                          num_workers=self.hparams.num_workers, collate_fn=utils.collate_fn)


# datamodel = RoofDetectionDataModule(ann_folder="annotations/2classes")
# num_classes, num_domains = datamodel.num_classes, datamodel.num_domains
model = FasterRCNN(ann_folder="annotations/2classes",)

trainer = L.Trainer(
    accelerator="auto",
    devices=1,
    max_epochs=5,
    enable_progress_bar=True,
    num_sanity_val_steps = 0
)
trainer.fit(model)

