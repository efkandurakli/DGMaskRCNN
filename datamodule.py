
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from dataset import get_dataset
import presets
import utils

def get_transform(is_train, data_augmentation="hflip", backend="pil", use_v2=False):
    if is_train:
        return presets.DetectionPresetTrain(
            data_augmentation=data_augmentation, backend=backend, use_v2=use_v2
        )
    else:
        return presets.DetectionPresetEval(backend=backend, use_v2=use_v2)

class RoofDetectionDataModule(pl.LightningDataModule):
    def __init__(self, data_path = "datasets/DFC2023/track1",
                 data_augmentation="hflip", backend="pil",
                 use_v2=False, batch_size = 2, num_workers = 4):
        
        super().__init__()

        self.data_path = data_path
        self.data_augmentation = data_augmentation
        self.backend = backend
        self.use_v2 = use_v2
        self.batch_size = batch_size
        self.num_workers = num_workers

        
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = get_dataset(self.data_path, 
                                    transforms=get_transform(True, data_augmentation=self.data_augmentation,backend=self.backend, use_v2=self.use_v2), 
                                    ann_folder="annotations/1classes", image_set='train')

            self.val_dataset = get_dataset(self.data_path, 
                                    transforms=get_transform(False, data_augmentation=self.data_augmentation,backend=self.backend, use_v2=self.use_v2), 
                                    ann_folder="annotations/1classes", image_set='val')

        if stage == "test":
            self.test_dataset = get_dataset(self.data_path, 
                                transforms=get_transform(False, data_augmentation=self.data_augmentation,backend=self.backend, use_v2=self.use_v2), 
                                ann_folder="annotations/1classes", image_set='test')


    def train_dataloader(self):
        train_sampler = torch.utils.data.RandomSampler(self.train_dataset)
        return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=train_sampler, 
                          num_workers=self.num_workers, collate_fn=utils.collate_fn)

    def val_dataloader(self):
        val_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        return DataLoader(self.train_dataset, batch_size=1, sampler=val_sampler, 
                          num_workers=self.num_workers, collate_fn=utils.collate_fn)


    def test_dataloader(self):
        test_sampler = torch.utils.data.SequentialSampler(self.test_dataset)
        return DataLoader(self.train_dataset, batch_size=1, sampler=test_sampler, 
                          num_workers=self.num_workers, collate_fn=utils.collate_fn)
