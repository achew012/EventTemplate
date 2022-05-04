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
        self.model = EventGraph(cfg.num_event_types, cfg.num_entity_types,  cfg.num_rels, cfg.n_hidden, cfg.emb_dim)
        self.remove = RemoveIsolatedNodes()
        self.metrics = {}

    def forward(self, batch):

        sub_g, bfs_list, sample_index, g = batch

        # torch_geometric.Batch -> torch_geometric.HeteroData
        sub_g = sub_g.to_data_list()[0]
        g = g.to_data_list()[0]

        g = g.to_homogeneous()
        sub_g = sub_g.to_homogeneous()

        # the difference between sub_g and g is that all the edges connected to the sampled event, events
        # afterwards, and their arguments in sub_g are removed. So there is a bunch of isolated nodes in sub_g.
        sub_g = self.remove(sub_g)

        for index, e in enumerate(bfs_list):
            bfs_list[index] = e.item()

        output = self.model(sub_g, bfs_list, sample_index)

        # get the real type of the event to be predicted
        target = g.x[bfs_list[sample_index]]
        target = target.long() - 1

        loss = nn.CrossEntropyLoss()(output, target)
        return loss, output

    def training_step(self, batch, batch_nb):
        """Call the forward pass then return loss"""
        loss, output = self.forward(*batch)
        return {"loss": loss, 'output': output}

    def training_epoch_end(self, outputs):
        total_loss = []
        for batch in outputs:
            total_loss.append(batch["loss"])
        self.log("train_loss", sum(total_loss) / len(total_loss))

    def eval_step(self, batch):
        batch_correct=0
        for test_graph in batch:
            sub_g, bfs_list, sample_index, g = test_graph
            ipdb.set_trace()
            #sub_g = sub_g.to_data_list()[0]
            #g = g.to_data_list()[0]
            g = g.to_homogeneous()
            sub_g = sub_g.to_homogeneous()
            sub_g = self.remove(sub_g)
            for index, e in enumerate(bfs_list):
                bfs_list[index] = e.item()
            target = g.x[bfs_list[sample_index]]
            target = target.int() - 1
            pred = self.model.test(sub_g, target)
            if pred:
                batch_correct += 1
        return batch_correct

    def validation_step(self, batch, batch_nb, dataloader_idx):        
        """Call the forward pass then return loss"""
        batch_matches = self.eval_step(batch)        
        return {dataloader_idx: {batch_nb: batch_matches}}

    def validation_epoch_end(self, outputs):
        hit_rate = []
        loss = []

        ipdb.set_trace()

        for dataloader in outputs:
            for batch in dataloader:
                batch_hit_k_normalized = batch["hit_k_normalized"]
                batch_loss = batch["loss"]
                loss.append(batch_loss)
                if batch_hit_k_normalized:
                    hit_rate.append(batch_hit_k_normalized)
        self.log("val_loss", torch.tensor(loss).mean())

    def test_step(self, batch, batch_nb, dataloader_idx):
        """Call the forward pass then return loss"""
        loss, output = self(batch)
        pred = output.max(dim=1)[1]
        return {"loss": loss, "batch": batch, "pred": pred}

    def test_epoch_end(self, outputs):
        correct = 0
        num_batches = len(outputs)
        for dataloader in outputs:
            for batch in dataloader:
                pred = batch["pred"]
                correct += pred.eq(batch['batch'].y.view(-1)).sum().item()
        self.log("accuracy", correct / (num_batches*self.cfg.batch_size))

    def configure_optimizers(self):
        """Configure the optimizer and the learning rate scheduler"""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, "min", verbose=True
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
