#!/usr/bin/env python
# coding: utf-8

import os, random
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchmetrics import Accuracy
import copy
import math
from snn_signgd.pretty_printer import print
from munch import Munch

class ImageClassification(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model = config.model()

        self.val_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        
        self.test_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = F.nll_loss(logits, y) # CrossEntropy = log_softmax + nll_loss
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        self.log("valid_loss", loss, prog_bar=True)
        self.log("valid_accuracy", self.val_accuracy, prog_bar=True)
        self.log("performance", self.val_accuracy, on_step=True, on_epoch=True)
        #print("Validation Step", loss, self.val_accuracy)

    def test_step(self, batch, batch_idx):
        x, y = batch

        logits = self(x)
        
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        
        self.test_accuracy.update(preds, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_accuracy", self.test_accuracy, prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = self.config.optimizer(self.model.parameters())
        lr_scheduler = self.config.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}   
        
    def setup(self,stage = None):
        if stage in ['fit', 'validate'] or stage is None:
            self.train_dataset = self.config.train_dataset(transform = self.config.preprocessors.train)
            self.valid_dataset = self.config.valid_dataset(transform = self.config.preprocessors.test)            
        elif stage in ['test', 'predict'] or stage is None:
            self.test_dataset = self.config.test_dataset(transform = self.config.preprocessors.test)        
        
        experimental_setup = self.config.setup(stage, self.config, self.model)
        
        self.model = experimental_setup
        return
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False, num_workers=4)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, drop_last=False, num_workers=4)