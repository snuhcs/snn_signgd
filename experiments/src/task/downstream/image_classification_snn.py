#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from ..image_classification import ImageClassification as BaseTask
import os

class ImageClassification(BaseTask):
    def __init__(self, config):
        super().__init__(config)

        self.timestamps = config.get('timestamps', [None]) # [] if it does not exist
        self.test_accuracys = nn.ModuleList(
            [ Accuracy(task="multiclass", num_classes=config.num_classes) for _ in self.timestamps ]
        )
        self.ref_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.true_ref_accuracy = Accuracy(task="multiclass", num_classes=config.num_classes)
        self.record = []

    def forward(self, x):
        x = self.model(x)
    
        if ('verbose' in self.config) and self.config.verbose: 
            print("SNN_output:", x)
            
        return F.log_softmax(x, dim=1)

    def test_step(self, batch, batch_idx):
        x, y = batch

        if None not in self.timestamps:
            logits, history, raw_output = self.forward_logging(
                x, timestamps = self.timestamps,
                measurement_device = self.measurement_device
            )
        else:
            logits = self(x)
            history = torch.stack([logits], dim = 0)
        
        if self.config.verbose: 
            reference_output = self.reference_model(x)
            print("ANN_output:", reference_output)
            
            preds = torch.argmax(reference_output, dim=1)
            self.ref_accuracy.update(preds, y)
            self.log("test_ref_acc", self.ref_accuracy, prog_bar=True)

            true_ref_output = self.true_ref_model(x)
            print("TRUE ANN_output:", true_ref_output)
            
            true_preds = torch.argmax(true_ref_output, dim=1)
            self.true_ref_accuracy.update(true_preds, y)
            self.log("[TRUE] test_ref_acc", self.true_ref_accuracy, prog_bar=True)
        
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(history, dim=2)

        self.log("test_loss", loss, prog_bar=True)
        
        for index, (accuracy, steps) in enumerate(zip(self.test_accuracys,self.timestamps)):
            accuracy.update(preds[index], y)
            self.log(f"({index}) test_acc_step[{steps}]", accuracy, prog_bar=True)

    def forward_logging(self, x, timestamps = [], measurement_device = None):
        y, history = self.model(x, timestamps = timestamps)

        if measurement_device is not None: 
            
            measurement_device.measurements[os.path.join('nn','input')] = x.detach().cpu().numpy()

            ann_output = self.reference_model(x)
            measurement_device.measurements[os.path.join('nn','output', 'ann')] = ann_output.detach().cpu().numpy()

            for timestep, output in zip(timestamps, history):
                measurement_device.measurements[os.path.join('nn','output', 'snn', str(timestep))] = output.detach().cpu().numpy()
        
        history = torch.stack(history, dim = 0)
        
        if self.config.verbose: print("SNN_output:", y)
            
        return F.log_softmax(y, dim=1), F.log_softmax(history, dim=2), y 
        
    def setup(self,stage = None):
        if stage in ['fit', 'validate'] or stage is None:
            self.train_dataset = self.config.train_dataset(transform = self.config.preprocessors.train)
            self.valid_dataset = self.config.valid_dataset(transform = self.config.preprocessors.test) 
            self.config.verbose = False           
        elif stage in ['test', 'predict'] or stage is None:
            self.test_dataset = self.config.test_dataset(transform = self.config.preprocessors.test)

        original_model = self.model
        experimental_setup = self.config.setup(stage, self.config, self.model)
        
        if isinstance(experimental_setup, tuple):
            self.model, self.reference_model, self.true_ref_model, self.measurement_device = experimental_setup
        else:
            self.model = experimental_setup
        return
