import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader 
from maskrcnn import maskrcnn_resnet50_fpn
from coco_eval import CocoEvaluator
from dataset import get_coco_api_from_dataset
from dataset import get_driving_dataset
import presets
import utils
from grl import GradientScalarLayer

def get_transform(is_train, data_augmentation="hflip", backend="pil", use_v2=False):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=data_augmentation, backend=backend, use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)


        
def prepare_masks(targets):
    masks = []
    for targets_per_image in targets:
        is_source = targets_per_image.get_field('is_source')
        mask_per_image = is_source.new_ones(1, dtype=torch.uint8) if is_source.any() else is_source.new_zeros(1, dtype=torch.uint8)
        masks.append(mask_per_image)
    return masks

def consistency_loss(img_feas, ins_fea, ins_labels, size_average=True):
    """
    Consistency regularization as stated in the paper
    `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
    L_cst = \sum_{i,j}||\frac{1}{|I|}\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
    """
    loss = []
    len_ins = ins_fea.size(0)
    intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
    for img_fea_per_level in img_feas:
        N, A, H, W = img_fea_per_level.shape
        img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
        img_feas_per_level = []
        assert N==2, \
            "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
        for i in range(N):
            img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
            img_feas_per_level.append(img_fea_mean)
        img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
        print(img_feas_per_level.shape, ins_fea.shape)
        loss_per_level = torch.abs(img_feas_per_level - ins_fea)
        loss.append(loss_per_level)
    loss = torch.cat(loss, dim=1)
    if size_average:
        return loss.mean()
    return loss.sum()

def compute_dg_img_loss(da_img, da_ins, da_img_consist, da_ins_consist, targets, da_ins_labels):
        da_img_flattened = []
        da_img_labels_flattened = []
        # for each feature level, permute the outputs to make them be in the
        # same format as the labels. Note that the labels are computed for
        # all feature levels concatenated, so we keep the same representation
        # for the image-level domain alignment
        for da_img_per_level in da_img:
            
            N, A, H, W = da_img_per_level.shape
            da_img_per_level = da_img_per_level.permute(0, 2, 3, 1)

            da_img_label_per_level = torch.zeros_like(da_img_per_level, dtype=torch.float32)
            da_img_label_per_level[targets, :] = 1

            da_img_per_level = da_img_per_level.reshape(N, -1)
            da_img_label_per_level = da_img_label_per_level.reshape(N, -1)
            
            da_img_flattened.append(da_img_per_level)
            da_img_labels_flattened.append(da_img_label_per_level)
            
        da_img_flattened = torch.cat(da_img_flattened, dim=1)
        da_img_labels_flattened = torch.cat(da_img_labels_flattened, dim=1)


        da_img_loss = F.binary_cross_entropy_with_logits(
            da_img_flattened, da_img_labels_flattened
        )

        da_ins_loss = F.binary_cross_entropy_with_logits(
            torch.squeeze(da_ins), da_ins_labels.type(torch.cuda.FloatTensor)
        )

        da_consist_loss = consistency_loss(da_img_consist, da_ins_consist, da_ins_labels, size_average=True)

        return da_img_loss, da_ins_loss, da_consist_loss



    
class DAImgHead(nn.Module):
    """
    Adds a simple Image-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
            USE_FPN (boolean): whether FPN feature extractor is used
        """
        super(DAImgHead, self).__init__()
        
        self.conv1_da = nn.Conv2d(in_channels, 512, kernel_size=1, stride=1)
        self.conv2_da = nn.Conv2d(512, 1, kernel_size=1, stride=1)

        for l in [self.conv1_da, self.conv2_da]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def forward(self, x):
        img_features = []
        for feature in x:
            t = F.relu(self.conv1_da(feature))
            img_features.append(self.conv2_da(t))
        return img_features
    
class DAInsHead(nn.Module):
    """
    Adds a simple Instance-level Domain Classifier head
    """

    def __init__(self, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(DAInsHead, self).__init__()


        self.fc1_da = nn.Linear(in_channels, 1024)
        self.fc2_da = nn.Linear(1024, 1024)
        self.fc3_da = nn.Linear(1024, 1)
        for l in [self.fc1_da, self.fc2_da]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.fc3_da.weight, std=0.05)
        nn.init.constant_(self.fc3_da.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.fc2_da(x))
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.fc3_da(x)
        return x
    
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
    
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
            cons_box = False,
            cons_mask = False,
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
        self.cons_box = cons_box
        self.cons_mask = cons_mask
        self.kwargs = kwargs

        self.save_hyperparameters()

        self.d1, self.num_classes = get_driving_dataset(self.data_path, 
                                transforms=get_transform(True, data_augmentation=self.data_augmentation,
                                                        backend=self.backend),
                                img_folder='images/10k/train', domain=0,
                                ann_folder=self.ann_folder, image_set="train")
        
        self.d2, _ = get_driving_dataset("/home/saad/efkan/Datasets/ACDC", 
                                transforms=get_transform(True, data_augmentation=self.data_augmentation,
                                                        backend=self.backend),
                                img_folder='rgb_anon', domain=1,
                                ann_folder='gt_detection', image_set="train")
        
        self.train_dataset = torch.utils.data.ConcatDataset([self.d1, self.d2])
        
        self.val_dataset, _ = get_driving_dataset("/home/saad/efkan/Datasets/Cityscapes", 
                                transforms=get_transform(False, data_augmentation=self.data_augmentation,
                                                        backend=self.backend),
                                img_folder="leftImg8bit/val", 
                                ann_folder="annotations", image_set="val")
                        
        self.test_dataset, _ = get_driving_dataset("/home/saad/efkan/Datasets/Cityscapes", 
                                transforms=get_transform(False, data_augmentation=self.data_augmentation,
                                                        backend=self.backend),
                                img_folder="leftImg8bit/val", 
                                ann_folder="annotations", image_set="val")


        self.num_domains = 2

        self.detector = maskrcnn_resnet50_fpn(num_classes=self.num_classes+1, **self.kwargs)
        
        self.detector.backbone.register_forward_hook(self.store_img_features)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_box_features)

        self.imghead = DAImgHead(256)
        self.inshead = DAInsHead(1024)

        self.grl_img = GradientScalarLayer(-1.0)
        self.grl_ins = GradientScalarLayer(-1.0)
        self.grl_img_consist = GradientScalarLayer(1.0)
        self.grl_ins_consist = GradientScalarLayer(1.0)

        self.awl = AutomaticWeightedLoss(3)

    
        """
        if self.image_dg:
            self.detector.backbone.register_forward_hook(self.store_img_features)
            self.imghead = DAImgHead(256)
        
        if self.box_dg:
            self.detector.roi_heads.box_head.register_forward_hook(self.store_box_features)
            self.boxhead = DGBoxHead(1024, self.num_domains)

        if self.mask_dg:
            self.detector.roi_heads.mask_head.register_forward_hook(self.store_mask_features)
            self.maskhead = DGMaskHead(self.num_domains)
        """
    
    def store_img_features(self, module, input, output):
       self.img_features = output

    def store_box_features(self, module, input, output):
        self.box_domains = input[1]
        self.box_features = output

    def store_mask_features(self, module, input, output):
        self.mask_domains = input[1]
        self.mask_features = output
      
    def forward(self, images):
        return self.detector(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict = self.detector(images, targets)

        img_domain_labels = torch.cat([target["domain"] for target in targets], dim=0)



        img_grl_fea = [self.grl_img(self.img_features[fea]) for fea in self.img_features]
        ins_grl_fea = self.grl_ins(self.box_features)
        img_grl_consist_fea = [self.grl_img_consist(self.img_features[fea]) for fea in self.img_features]
        ins_grl_consist_fea = self.grl_ins_consist(self.box_features)

        da_img_features = self.imghead(img_grl_fea)
        da_ins_features = self.inshead(ins_grl_fea)
        da_img_consist_features = self.imghead(img_grl_consist_fea)
        da_ins_consist_features = self.inshead(ins_grl_consist_fea)
        da_img_consist_features = [fea.sigmoid() for fea in da_img_consist_features]
        da_ins_consist_features = da_ins_consist_features.sigmoid()



        img_loss, ins_loss, da_consistency_loss = compute_dg_img_loss(da_img_features, da_ins_features, da_img_consist_features, da_ins_consist_features, img_domain_labels, self.box_domains)
    

        self.log("img_loss", img_loss)
        self.log("ins_loss", ins_loss)
        self.log("cons_loss", da_consistency_loss)

        dg_loss = self.awl(img_loss, ins_loss, da_consistency_loss)

        loss_dict.update({"dg_loss": dg_loss})
        

        """
        if self.image_dg:
            img_domain_labels = torch.cat([target["domain"] for target in targets], dim=0)
            img_domain_logits = self.imghead(self.img_features["0"])
            img_loss = F.cross_entropy(img_domain_logits, img_domain_labels)
            loss_dict.update({"img_dg_loss": img_loss})
        
        if self.box_dg:
            box_domain_logits = self.boxhead(self.box_features)
            box_dg_loss = F.cross_entropy(box_domain_logits, self.box_domains)
            loss_dict.update({"box_dg_loss": box_dg_loss})

        if self.mask_dg:
            mask_domain_logits = self.maskhead(self.mask_features)
            mask_dg_loss = F.cross_entropy(mask_domain_logits, self.mask_domains)
            loss_dict.update({"mask_dg_loss": mask_dg_loss})

        if  self.image_dg and self.box_dg and self.cons_box:
            cons_loss_box = F.mse_loss(box_domain_logits, img_domain_logits[0].repeat(box_domain_logits.shape[0],1))
            loss_dict.update({"cons_loss_box": cons_loss_box})

        if  self.image_dg and self.mask_dg and self.cons_mask:
            cons_loss_mask = F.mse_loss(mask_domain_logits, img_domain_logits[0].repeat(mask_domain_logits.shape[0],1))
            loss_dict.update({"cons_loss_mask": cons_loss_mask})

        """

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
        print("map50: ", map50)
        self.log("map@50", map50)

    def on_test_start(self):
        coco = get_coco_api_from_dataset(self.test_dataset)
        self.coco_evaluator = CocoEvaluator(coco, ["bbox", "segm"])

    def on_test_end(self):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()

        print(self.coco_evaluator.coco_eval["bbox"].stats[5:-1])



    def configure_optimizers(self):
        parameters = [p for p in self.parameters() if p.requires_grad]
        if self.opt.lower() == 'sgd':
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
