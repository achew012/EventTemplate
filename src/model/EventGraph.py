import math
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
#from RGCN import RGCNConv
from torch_geometric.nn import RGCNConv
import numpy as np


class EventGraph(nn.Module):

    def __init__(self, num_event_types, num_entity_types, num_rels, hidden_channels, embedding_dim, dropout=0.) -> None:
        super().__init__()

        self.num_event_types = num_event_types
        self.num_entity_types = num_entity_types
        self.num_rels_types = num_rels  # includes types of event-event, event-entity, entity-entity

        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.node_emb = nn.Embedding(num_event_types + num_entity_types, embedding_dim)
        self.rel_emb = nn.Embedding(num_rels, embedding_dim)

        self.conv = RGCNConv(embedding_dim, hidden_channels, num_rels)

        # the future section 3.3 Event Generation
        # after generating new events, write a function `get_arguments(events)` to add arguments
        # add EOG (end of graph), num_events = num_of_event_types + SOG (if needed) + EOG
        # self.event_lin = nn.Linear(embedding_dim, num_events)
        self.event_lin = nn.Linear(hidden_channels, num_event_types - 1)

        # the future section 3.6 Entity Relational Edge Generation
        # add virtual edges first
        self.rel_lin = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, num_rels)
        )

        self.reset()


    def reset(self):        
        self.node_emb.reset_parameters()
        self.rel_emb.reset_parameters()

        self.event_lin.reset_parameters()


    def forward(self, g, bfs_list, sample_index):
        """
        Given a train data batch, computes the forward pass.

        Args:
            g (torch_geometric.data.Data): The input graph
            bfs_list (list): event nodes' order of the input graph obtained by using Breadth-First Search
            sample_index (int): index of the event to be predicted in the bfs_list
        """
        # get index of event nodes
        event_idx = []
        for idx, type in enumerate(g.node_type):
            if type == 0:
                event_idx.append(idx)

        # get node embeddings according to their types
        feat_idx = g.x.int().T
        feat = self.node_emb(feat_idx)
        feat = feat.squeeze(0)

        h = F.relu(self.conv(feat, g.edge_index, g.edge_type))

        # paper 3.3, get current graph representation g_i
        #event_idx = np.array(event_idx)
        #event_idx = event_idx[bfs_list[:sample_index]]  # index of event nodes e(0) ~ e(i-1)
        event_repr = h[event_idx]  # embeddings of e(0) ~ e_(i-1)
        g_i = torch.mean(event_repr, 0)

        out = self.event_lin(g_i)
        out = out.unsqueeze(0)

        return out


    def test(self, g, target):
        event_idx = []
        for idx, type in enumerate(g.node_type):
            if type == 0:
                event_idx.append(idx)

        feat_idx = g.x.int().T
        feat = self.node_emb(feat_idx)
        feat = feat.squeeze(0)
        h = F.relu(self.conv(feat, g.edge_index, g.edge_type))
        event_repr = h[event_idx]  # embeddings of e(0) ~ e_(i-1)
        g_i = torch.mean(event_repr, 0)
        out = self.event_lin(g_i)
        out = out.unsqueeze(0)
        pred = torch.argmax(out)

        return pred == target
