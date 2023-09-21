import torch
import torchvision
import lightning as L

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
            num_classes: int = 1,
            lr: float = 0.02,
            momentum: float = 0.9,
            weight_decay: float = 1e-4,
            **kwargs,
        ):
        super().__init__()
        self.save_hyperparameters()
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=self.hparams.num_classes, **kwargs)

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def on_train_start(self):
        pass

    def on_train_end(self):
        pass

    def on_validation_start(self):
        pass

    def on_validation_end(self):
        pass

    def on_test_start(self):
        pass

    def on_test_end(self):
        pass

    def configure_optimizers(self):
        parameters = [p for p in self.detector.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay)
        return optimizer

