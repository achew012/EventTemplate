import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
#from RGCN import RGCNConv
from torch_geometric.nn import RGCNConv
import numpy as np


class SingleProp(nn.Module):

    def __init__(self, num_event_types, num_entity_types, num_rels, hidden_channels, embedding_dim, dropout=0.) -> None:
        super().__init__()

        self.num_event_types = num_event_types
        self.num_entity_types = num_entity_types
        # includes types of event-event, event-entity, entity-entity
        self.num_rels_types = num_rels

        self.hidden_channels = hidden_channels
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        self.node_emb = nn.Embedding(
            num_event_types + num_entity_types, embedding_dim)
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

    def forward(self, g):
        """
        Given a train data batch, computes the forward pass for each graph in this batch

        Args:
            g (torch_geometric.data.Batch): A data object describing a batch of graphs as one big (disconnected) graph.
        """
        # get index of event nodes in each graph
        # g.num_graphs returns number of graphs in a batch
        events = []
        for graph_n in range(g.num_graphs):
            event_idx = []
            for idx, b in enumerate(g.batch):
                if b == graph_n and g.node_type[idx] == 0:
                    event_idx.append(idx)
            events.append(event_idx)

        # get node embeddings according to their types
        # g.x stores the type id of each node
        feat_idx = g.x.int().T
        # map feature index to feature embeddings
        feat = self.node_emb(feat_idx)
        feat = feat.squeeze(0)

        # apply rgcn convolution layer to feat embeddings
        h = F.relu(self.conv(feat, g.edge_index, g.edge_type))

        # paper 3.3, get current graph representation g_i for each graph in a batch
        for idx, event_idx in enumerate(events):
            event_repr = h[event_idx]
            if idx == 0:
                g_i = torch.mean(event_repr, 0)
                g_i = g_i.unsqueeze(0)
            else:
                g_i_temp = torch.mean(event_repr, 0)
                g_i_temp = g_i_temp.unsqueeze(0)
                g_i = torch.cat((g_i, g_i_temp))
        out = self.event_lin(g_i)
        return out

    def test(self, g, target):
        events = []
        for graph_n in range(g.num_graphs):
            event_idx = []
            for idx, b in enumerate(g.batch):
                if b == graph_n and g.node_type[idx] == 0:
                    event_idx.append(idx)
            events.append(event_idx)
        feat_idx = g.x.int().T
        feat = self.node_emb(feat_idx)
        feat = feat.squeeze(0)
        h = F.relu(self.conv(feat, g.edge_index, g.edge_type))
        for idx, event_idx in enumerate(events):
            event_repr = h[event_idx]
            if idx == 0:
                g_i = torch.mean(event_repr, 0)
                g_i = g_i.unsqueeze(0)
            else:
                g_i_temp = torch.mean(event_repr, 0)
                g_i_temp = g_i_temp.unsqueeze(0)
                g_i = torch.cat((g_i, g_i_temp))
        out = self.event_lin(g_i)
        pred = torch.argmax(out, 1)
        return pred
