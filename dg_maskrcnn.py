import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader 
from gradient_scalar_layer import GradientScalarLayer
from maskrcnn import maskrcnn_resnet50_fpn
from coco_eval import CocoEvaluator
from dataset import get_coco_api_from_dataset
from dataset import get_dataset
import presets
import utils


DG_IMG_GRL_WEIGHT = 0.1
DG_BOX_GRL_WEIGHT = 0.1
DG_MASK_GRL_WEIGHT = 0.1


def get_transform(is_train, data_augmentation="hflip", backend="pil", use_v2=False):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=data_augmentation, backend=backend, use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)

class DGImgHead(nn.Module):
    def __init__(self, in_channels, num_domains):
        super(DGImgHead, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        self.fc1 = nn.Linear(53294, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_domains)
        self.softmax = nn.Softmax(dim=1)

        for l in [self.conv1, self.conv2]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

        for l in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

        nn.init.normal_(self.classifier.weight, std=0.05)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1(feature))
            img_features.append(self.conv2(t))

        img_features_flattened = []
        for img_feature in img_features:
            N = img_feature.shape[0]
            img_feature = img_feature.permute(0, 2, 3, 1)
            img_feature = img_feature.reshape(N, -1)
            img_features_flattened.append(img_feature)

        img_features_flattened = torch.cat(img_features_flattened, dim=1)

        img_features_flattened = F.relu(self.fc1(img_features_flattened))
        img_features_flattened = F.dropout(img_features_flattened, p=0.5)

        img_features_flattened = F.relu(self.fc2(img_features_flattened))
        img_features_flattened = F.dropout(img_features_flattened, p=0.5)

        img_features_flattened = F.relu(self.fc3(img_features_flattened))
        img_features_flattened = F.dropout(img_features_flattened, p=0.5)

        img_features_flattened = self.classifier(img_features_flattened)


        return self.softmax(img_features_flattened)
    
class DGInsHead(nn.Module):
    def __init__(self, in_channels, num_domains):
        super(DGInsHead, self).__init__()
        self.fc1 = nn.Linear(in_channels, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_domains)
        self.softmax = nn.Softmax(dim=1)

        for l in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.05)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.5)

        x = F.relu(self.fc3(x))
        x = F.dropout(x, p=0.5)

        x = self.classifier(x)

        return self.softmax(x)
    
class DGMaskRCNN(L.LightningModule):
    def __init__(
            self,
            data_path = "/home/duraklefkan/workspace/Datasets/DFC2023/track1",
            ann_folder = "annotations/1classes",
            data_augmentation="hflip",
            backend="pil",
            batch_size=2,
            num_workers = 4,
            lr = 0.01,
            momentum = 0.9,
            weight_decay = 1e-4,
            opt = 'sgd',
            lr_steps = [16, 22],
            lr_gamma = 0.1,
            image_dg = False,
            box_dg = False,
            mask_dg = False,
            **kwargs
    ):
        super().__init__()

        self.data_path = data_path
        self.ann_folder = ann_folder
        self.data_augmentation = data_augmentation
        self.backend = backend
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.opt = opt
        self.lr_steps = lr_steps
        self.lr_gamma = lr_gamma
        self.image_dg = image_dg
        self.box_dg = box_dg
        self.mask_dg = mask_dg
        self.kwargs = kwargs

        self.save_hyperparameters()

        self.train_dataset, self.num_classes, self.num_domains = get_dataset(self.data_path, 
                                transforms=get_transform(True, data_augmentation=self.data_augmentation,
                                                        backend=self.backend), 
                                ann_folder=self.ann_folder, image_set="train")
        
        self.val_dataset, _, _ = get_dataset(self.data_path, 
                                transforms=get_transform(False, data_augmentation=self.data_augmentation,
                                                        backend=self.backend), 
                                ann_folder=self.ann_folder, image_set="val")
                        

        self.test_dataset, self.num_classes, self.num_domains = get_dataset(self.data_path, 
                                transforms=get_transform(False, data_augmentation=self.data_augmentation,
                                                        backend=self.backend), 
                                ann_folder=self.ann_folder, image_set="test")
        
        self.detector = maskrcnn_resnet50_fpn(num_classes=self.num_classes+1, **self.kwargs)


        if self.image_dg:
            self.detector.backbone.register_forward_hook(self.store_img_features)
            self.grl_img = GradientScalarLayer(-1.0*DG_IMG_GRL_WEIGHT)
            self.imghead = DGImgHead(256, self.num_domains)
        
        if self.box_dg:
            self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)
            self.grl_box = GradientScalarLayer(-1.0*DG_BOX_GRL_WEIGHT)
            self.boxhead = DGInsHead(1024, self.num_domains)

        if self.mask_dg:
            self.detector.roi_heads.mask_head.register_forward_hook(self.store_mask_features)
            self.grl_mask = GradientScalarLayer(-1.0*DG_MASK_GRL_WEIGHT)
            self.maskhead = DGInsHead(256*14*14, self.num_domains)
    
    def store_img_features(self, module, input, output):
       self.img_features = output

    def store_box_features(self, module, input, output):
        self.box_domains = input[1]
        self.box_features = output

    def store_mask_features(self, module, input, output):
        self.mask_domains = input[1]
        self.mak_features = output
      
    def forward(self, images):
        return self.detector(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = self.detector(images, targets)

        if self.image_dg:
            img_domain_labels = torch.cat([target["domain"] for target in targets], dim=0)
            img_grl_fea = [self.grl_img(self.img_features[fea]) for fea in self.img_features]
            img_domain_logits = self.imghead(img_grl_fea)
            img_loss = F.cross_entropy(img_domain_logits, img_domain_labels)
            loss_dict.update({"img_loss": img_loss})
        
        if self.box_dg:
            box_grl_fea = self.grl_box(self.box_features)
            box_domain_logits = self.boxhead(box_grl_fea)
            box_dg_loss = F.cross_entropy(box_domain_logits, self.box_domains)
            loss_dict.update({"box_dg_loss": box_dg_loss})

        if self.mask_dg:
            mask_fetures = self.mak_features.flatten(start_dim=1)
            mask_grl_fea = self.grl_mask(mask_fetures)
            mask_domain_logits = self.maskhead(mask_grl_fea)
            mask_dg_loss = F.cross_entropy(mask_domain_logits, self.mask_domains)
            loss_dict.update({"mask_dg_loss": mask_dg_loss})

        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)
        self.log_dict(loss_dict)

        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        images = list(img for img in images)
        outputs = self(images)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

    def test_step(self, batch, batch_idx):
        images, targets = batch
        images = list(img for img in images)
        outputs = self(images)
        outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

    def on_validation_epoch_start(self):
        coco = get_coco_api_from_dataset(self.val_dataset)
        self.coco_evaluator = CocoEvaluator(coco, ["bbox", "segm"])

    def on_validation_epoch_end(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        map50 = self.coco_evaluator.coco_eval['segm'].stats[1]
        self.log("map@50", map50)

    def on_test_start(self):
        coco = get_coco_api_from_dataset(self.test_dataset)
        self.coco_evaluator = CocoEvaluator(coco, ["bbox", "segm"])

    def on_test_end(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

    def configure_optimizers(self):
        parameters = [p for p in self.detector.parameters() if p.requires_grad]
        if self.opt.lower():
            optimizer = torch.optim.SGD(parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            optimizer = torch.optim.AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)
        
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_steps, gamma=self.lr_gamma)

        return [optimizer], [lr_scheduler]
    
    def train_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, 
                          num_workers=self.num_workers, collate_fn=utils.collate_fn)

    def val_dataloader(self):
        val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        return DataLoader(self.val_dataset, batch_size=1, sampler=val_sampler, 
                          num_workers=self.num_workers, collate_fn=utils.collate_fn)
    
    def test_dataloader(self):
        val_sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        return DataLoader(self.test_dataset, batch_size=1, sampler=val_sampler, 
                          num_workers=self.num_workers, collate_fn=utils.collate_fn)
