import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader 
from maskrcnn import maskrcnn_resnet50_fpn
from coco_eval import CocoEvaluator
from dataset import get_coco_api_from_dataset
from dataset import get_dataset
import presets
import utils
from torch.autograd import Function


def get_transform(is_train, data_augmentation="hflip", backend="pil", use_v2=False):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=data_augmentation, backend=backend, use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)


class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)

class _InstanceDA(nn.Module):
    def __init__(self, num_domains):
        super(_InstanceDA,self).__init__()
        self.num_domains = num_domains
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_domains)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x
    
class _InsClsPrime(nn.Module):
    def __init__(self, num_cls):
        super(_InsClsPrime,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=grad_reverse(x)
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x
    
class _InsCls(nn.Module):
    def __init__(self, num_cls):
        super(_InsCls,self).__init__()
        self.num_cls = num_cls
        self.dc_ip1 = nn.Linear(1024, 512)
        self.dc_relu1 = nn.ReLU()
        #self.dc_drop1 = nn.Dropout(p=0.5)

        self.dc_ip2 = nn.Linear(512, 256)
        self.dc_relu2 = nn.ReLU()
        #self.dc_drop2 = nn.Dropout(p=0.5)

        self.classifer=nn.Linear(256,self.num_cls)
        

    def forward(self,x):
        x=self.dc_relu1(self.dc_ip1(x))
        x=self.dc_ip2(x)
        x=torch.sigmoid(self.classifer(x))

        return x
    
class _ImageDA(nn.Module):
    def __init__(self,dim,num_domains):
        super(_ImageDA,self).__init__()
        self.dim=dim  # feat layer          256*H*W for vgg16
        self.num_domains = num_domains
        self.Conv1 = nn.Conv2d(self.dim, 256, kernel_size=3, stride=4)
        self.Conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.Conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=4)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, self.num_domains)
        self.reLu=nn.ReLU(inplace=False)
        

 
        
    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.Conv1(x))
        x=self.reLu(self.Conv2(x))
        x=self.reLu(self.Conv3(x))
        x=self.reLu(self.Conv4(x))
        x=self.flatten(x)
        x=self.reLu(self.linear1(x))
        x=torch.sigmoid(self.linear2(x))
        
        return x

class DGFasterCNN(L.LightningModule):
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
            ins_dg = False,
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
        self.ins_dg = ins_dg
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
                        

        self.test_dataset, _, _ = get_dataset(self.data_path, 
                                transforms=get_transform(False, data_augmentation=self.data_augmentation,
                                                        backend=self.backend), 
                                ann_folder=self.ann_folder, image_set="test")
        
        self.detector = maskrcnn_resnet50_fpn(num_classes=self.num_classes+1, **self.kwargs)


        self.ImageDA = _ImageDA(256, self.num_domains)
        self.InsDA = _InstanceDA(self.num_domains)
        self.InsCls = nn.ModuleList([_InsCls(self.num_classes+1) for i in range(self.num_domains)])
        self.InsClsPrime = nn.ModuleList([_InsClsPrime(self.num_classes+1) for i in range(self.num_domains)])

        self.base_lr = 1e-5 #Original base lr is 1e-4
        self.momentum = 0.9
        self.weight_decay=0.0001

        self.detector.backbone.register_forward_hook(self.store_img_features)
        self.detector.roi_heads.box_head.register_forward_hook(self.store_ins_features)

        self.mode = 0
        self.sub_mode = 0

    def store_img_features(self, module, input, output):
       self.img_features = output

    def store_ins_features(self, module, input, output):
            self.box_domains = input[1]
            self.box_labels = input[2]
            self.box_features = output
      
    def forward(self, images):
        return self.detector(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        images = list(image for image in images)
        targets = [{k: v if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss = None
        img_domain_labels = torch.cat([target["domain"] for target in targets], dim=0)

        if(self.mode == 0):
            loss_dict = self.detector(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            if(self.sub_mode == 0):
                self.mode = 1
                self.sub_mode = 1
            elif(self.sub_mode == 1):
                self.mode = 2
                self.sub_mode = 2
            elif(self.sub_mode == 2):
                self.mode = 3
                self.sub_mode = 3
            elif(self.sub_mode == 3):
                self.mode = 4
                self.sub_mode = 4  
            else:
                self.sub_mode = 0
                self.mode = 0
        
        elif(self.mode == 1):
            loss_dict = {}
            self.detector(images, targets)
            ImgDA_scores = self.ImageDA(self.img_features['0'])
            loss_dict['DA_img_loss'] = 0.5*F.cross_entropy(ImgDA_scores, img_domain_labels)
            IDA_out = self.InsDA(self.box_features)
            loss_dict['DA_ins_loss'] = F.cross_entropy(IDA_out, self.box_domains)
            loss_dict['Cst_loss'] = F.mse_loss(IDA_out, ImgDA_scores[0].repeat(IDA_out.shape[0],1))
            loss = sum(loss for loss in loss_dict.values())
            self.mode = 0
        
        elif(self.mode == 2):
            loss_dict = {}
            for index in range(len(self.InsCls)):
                for param in self.InsCls[index].parameters(): param.requires_grad = True

            with torch.no_grad():
                self.detector(images, targets)
            
            temp_losses = []
            for index in range(len(images)):
                d = img_domain_labels[index].item()
                cls_scores = self.InsCls[d](self.box_features)
                temp_losses.append(F.cross_entropy(cls_scores, torch.cat(self.box_labels, dim=0)))

            loss_dict['cls'] = 0.05*(torch.mean(torch.stack(temp_losses)))
            loss = sum(loss for loss in loss_dict.values())

            self.mode = 0
        
        elif(self.mode == 3):
            self.detector(images, targets)
            loss_dict = {}
            temp_losses = []
            for index in range(len(images)):
                d = img_domain_labels[index].item()
                cls_scores = self.InsClsPrime[d](self.box_features)
                temp_losses.append(F.cross_entropy(cls_scores, torch.cat(self.box_labels, dim=0)))

            loss_dict['cls_prime'] = 0.0001*(torch.mean(torch.stack(temp_losses)))
            loss = sum(loss for loss in loss_dict.values())

            self.mode = 0

        else:
            for index in range(len(self.InsCls)):
                for param in self.InsCls[index].parameters(): param.requires_grad = False
            
            self.detector(images, targets)
            temp_losses = []
            consis_loss = []
            loss_dict = {}
            for index in range(len(images)):
                d = img_domain_labels[index].item()
                temp = []
                for i in range(len(self.InsCls)):
                    if(i != d):
                        cls_scores = self.InsCls[i](self.box_features)
                        temp.append(cls_scores)
                        temp_losses.append(F.cross_entropy(cls_scores, torch.cat(self.box_labels, dim=0)))
                consis_loss.append(torch.mean(torch.abs(torch.stack(temp, dim=0) - torch.mean(torch.stack(temp, dim=0), dim=0))))
            
            loss_dict['cls'] = 0.05*(torch.mean(torch.stack(temp_losses)))
            loss = sum(loss for loss in loss_dict.values())
            self.mode = 0
            self.sub_mode = 0

        self.log("train_loss", loss)
        return loss

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
        
        optimizer = torch.optim.Adam([{'params': self.detector.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.ImageDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsDA.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsCls.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay },
                                    {'params': self.InsClsPrime.parameters(), 'lr': self.base_lr, 'weight_decay': self.weight_decay}
                                      ],) 
      
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
