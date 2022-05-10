import os
from torch import nn
import torch
from typing import List, Any, Dict

#from common.utils import *
import pytorch_lightning as pl

import torch
import torch.nn.functional as F

from clearml import Dataset as ClearML_Dataset
from torch_geometric.transforms import RemoveIsolatedNodes
from common.utils import *
from model.EventGraph import EventGraph
import ipdb


class GraphEmbedding(pl.LightningModule):
    """Pytorch Lightning module. It wraps up the model, data loading and training code"""

    def __init__(self, cfg, task):
        """Loads the model, the tokenizer and the metric."""
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.clearml_logger = self.task.get_logger()
        self.model = EventGraph(
            cfg.num_event_types, cfg.num_entity_types,  cfg.num_rels, cfg.n_hidden, cfg.emb_dim)
        self.remove = RemoveIsolatedNodes()
        self.metrics = {}

    def forward(self, batch):

        sub_g, target = batch

        sub_g = self.remove(sub_g)

        output = self.model(sub_g)

        # get the real type of the event to be predicted
        target = target.T
        target = target.long() - 1
        target = target.squeeze(0)

        # get the real type of the event to be predicted
        loss = nn.CrossEntropyLoss()(output, target)
        return loss, output

    def training_step(self, batch: List, batch_nb: int):
        """Call the forward pass then return loss"""
        #should only b a single element in batch list
        for sample in batch:
            loss, output = self(sample)
        return {"loss": loss, 'output': output}

    def training_epoch_end(self, outputs: List):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("train_loss", sum(total_loss) / len(total_loss))

    def eval_step(self, batch: List):
        n = 0
        batch_correct = 0
        for sample in batch:
            sub_g, target = sample
            n += len(target)  # number of graphs in 1 batch
            sub_g = self.remove(sub_g)
            target = target.T
            target = target.long() - 1
            target = target.squeeze(0)
            preds = self.model.test(sub_g, target)
            ipdb.set_trace()
            batch_correct += torch.sum(preds).detach().item()  # count the number of 'True' in preds
        return batch_correct/n

    def validation_step(self, batch:List, batch_nb: int, dataloader_idx: int):
        """Call the forward pass then return loss"""
        batch_accuracy = self.eval_step(batch)
        return {'batch_accuracy': batch_accuracy}

    def validation_epoch_end(self, outputs: List):
        best_accuracy = 0
        for idx, dataload in enumerate(outputs):
            accuracy = 0
            num_batches = len(dataload)
            for batch_output in dataload:
                accuracy += batch_output["batch_accuracy"]

            accuracy=accuracy/num_batches

            if accuracy>best_accuracy:
                best_accuracy=accuracy

        self.log(f"best_val_accuracy", best_accuracy)

    def test_step(self, batch: List, batch_nb: int, dataloader_idx: int):
        """Call the forward pass then return loss"""
        batch_accuracy = self.eval_step(batch)
        return {'batch_accuracy': batch_accuracy}

    def test_epoch_end(self, outputs: List):
        best_accuracy = 0
        for idx, dataload in enumerate(outputs):
            accuracy = 0
            num_batches = len(dataload)
            for batch_output in dataload:
                accuracy += batch_output["batch_accuracy"]

            accuracy=accuracy/num_batches

            if accuracy>best_accuracy:
                best_accuracy=accuracy

        self.log(f"best_test_accuracy", best_accuracy)

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "max", verbose=True
                ),
                "monitor": "best_val_accuracy",
                "frequency": 1,
            },
        }
