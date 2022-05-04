from torch_geometric.data import Dataset
from torch_geometric.data import HeteroData
import torch
import os
import random
import ipdb


def get_subgraph(bfs_list, sample_index, event_event, event_entity, entity_entity, event_type, entity_type):
    event_del = bfs_list[sample_index:]  # events to be deleted
    for key in event_event.keys():
        for i, src_node in enumerate(event_event[key]['src']):
            if src_node in event_del:
                del event_event[key]['src'][i]
                del event_event[key]['dst'][i]
        for i, dst_node in enumerate(event_event[key]['dst']):
            if dst_node in event_del:
                del event_event[key]['src'][i]
                del event_event[key]['dst'][i]
    entity_del = []  # entities to be deleted
    for key in event_entity.keys():
        for i, src_node in enumerate(event_entity[key]['src']):
            if src_node in event_del:
                del event_entity[key]['src'][i]
                if event_entity[key]['dst'][i] not in entity_del:
                    entity_del.append(event_entity[key]['dst'][i])
                del event_entity[key]['dst'][i]
    for key in entity_entity.keys():
        for i, src_node in enumerate(entity_entity[key]['src']):
            if src_node in entity_del:
                del entity_entity[key]['src'][i]
                del entity_entity[key]['dst'][i]
        for i, dst_node in enumerate(entity_entity[key]['dst']):
            if dst_node in event_del:
                del entity_entity[key]['src'][i]
                del entity_entity[key]['dst'][i]

    sub_g = HeteroData()
    sub_g['event'].x = torch.tensor([event_type], dtype=torch.float).T
    sub_g['entity'].x = torch.tensor([entity_type], dtype=torch.float).T
    for key in event_event.keys():
        src = torch.tensor(event_event[key]['src'], dtype=torch.long)
        dst = torch.tensor(event_event[key]['dst'], dtype=torch.long)
        sub_g['event', key, 'event'].edge_index = torch.stack((src, dst))
    for key in event_entity.keys():
        src = torch.tensor(event_entity[key]['src'], dtype=torch.long)
        dst = torch.tensor(event_entity[key]['dst'], dtype=torch.long)
        sub_g['event', key, 'entity'].edge_index = torch.stack((src, dst))
    for key in entity_entity.keys():
        src = torch.tensor(entity_entity[key]['src'], dtype=torch.long)
        dst = torch.tensor(entity_entity[key]['dst'], dtype=torch.long)
        sub_g['entity', key, 'entity'].edge_index = torch.stack((src, dst))

    return sub_g


def get_single_graph(path, mode, ratio):
    g = HeteroData()  # create heterogeneous graph

    bfs_list = []
    event_type = []
    entity_type = []
    with open(path + 'event_type.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num = line.strip().split("\t")[1]
            event_type.append(int(num))
    with open(path + 'entity_type.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            num = line.strip().split("\t")[1]
            entity_type.append(int(num))
    g['event'].x = torch.tensor([event_type], dtype=torch.float).T
    g['entity'].x = torch.tensor([entity_type], dtype=torch.float).T

    adj_dict = {}  # create adjacency dict to implement BFS
    event_event_dict = {}
    with open(path + 'event_event.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            triplet = line.strip().split("\t")
            src, rel, dst = triplet
            if rel not in event_event_dict.keys():
                event_event_dict[rel] = {'src': [], 'dst': []}
            event_event_dict[rel]['src'].append(int(src))
            event_event_dict[rel]['dst'].append(int(dst))

        # relation "0" denotes the temporal relation "before" between events
        # if relation "0" doesn't exist, return "None"
        if '0' in event_event_dict.keys():
            # get all event nodes that are not the dst of relation(event, 'before', event)
            root_nodes = []
            for s in event_event_dict['0']['src']:
                if s not in event_event_dict['0']['dst'] and s not in root_nodes:
                    root_nodes.append(s)

            # get all event nodes that are not the dst of relation(event, 'after', event)
            leaf_nodes = []
            for s in event_event_dict['1']['src']:
                if s not in event_event_dict['1']['dst'] and s not in leaf_nodes:
                    leaf_nodes.append(s)

            # add SOG before root_nodes
            for r in root_nodes:
                event_event_dict['0']['src'].append(0)
                event_event_dict['0']['dst'].append(r)
                event_event_dict['1']['src'].append(r)
                event_event_dict['1']['dst'].append(0)
            # add EOG after leaf_nodes
            for l in leaf_nodes:
                event_event_dict['1']['src'].append(1)
                event_event_dict['1']['dst'].append(l)
                event_event_dict['0']['src'].append(l)
                event_event_dict['0']['dst'].append(1)

            # get adjacency dict of relation '0'('before')
            for index, s in enumerate(event_event_dict['0']['src']):
                if s not in adj_dict.keys():
                    adj_dict[s] = []
                adj_dict[s].append(event_event_dict['0']['dst'][index])

            # use adjacency dict to implement BFS
            # 0 is the id of SOG, 1 is the id of EOG
            bfs_list.append(0)
            for node in bfs_list:
                if node in adj_dict.keys():
                    for n in adj_dict[node]:
                        if n not in bfs_list and n is not 1:
                            bfs_list.append(n)
            bfs_list.append(1)

            for key in event_event_dict.keys():
                src = torch.tensor(event_event_dict[key]['src'], dtype=torch.long)
                dst = torch.tensor(event_event_dict[key]['dst'], dtype=torch.long)
                g['event', key, 'event'].edge_index = torch.stack((src, dst))
        else:
            return None, None, None, None

    event_entity_dict = {}
    with open(path + 'event_entity.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            triplet = line.strip().split("\t")
            src, rel, dst = triplet
            if rel not in event_entity_dict.keys():
                event_entity_dict[rel] = {'src': [], 'dst': []}
            event_entity_dict[rel]['src'].append(int(src))
            event_entity_dict[rel]['dst'].append(int(dst))
        for key in event_entity_dict.keys():
            src = torch.tensor(event_entity_dict[key]['src'], dtype=torch.long)
            dst = torch.tensor(event_entity_dict[key]['dst'], dtype=torch.long)
            g['event', key, 'entity'].edge_index = torch.stack((src, dst))

    entity_entity_dict = {}
    with open(path + 'entity_entity.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            triplet = line.strip().split("\t")
            src, rel, dst = triplet
            if rel not in entity_entity_dict.keys():
                entity_entity_dict[rel] = {'src': [], 'dst': []}
            entity_entity_dict[rel]['src'].append(int(src))
            entity_entity_dict[rel]['dst'].append(int(dst))
        for key in entity_entity_dict.keys():
            src = torch.tensor(entity_entity_dict[key]['src'], dtype=torch.long)
            dst = torch.tensor(entity_entity_dict[key]['dst'], dtype=torch.long)
            g['entity', key, 'entity'].edge_index = torch.stack((src, dst))

    if mode == 'train':
        sample_index = random.randint(2, len(bfs_list)-1)
        sub_g = get_subgraph(bfs_list, sample_index, event_event_dict, event_entity_dict, entity_entity_dict, event_type, entity_type)
        return sub_g, bfs_list, sample_index, g
    elif mode == 'eval':
        sub_g_list = []
        bfs = []
        indices = []
        g_list = []
        if ratio == 0.:
            for i in range(2, len(bfs_list)):
                sub_g = get_subgraph(bfs_list, i, event_event_dict, event_entity_dict, entity_entity_dict, event_type, entity_type)
                sub_g_list.append(sub_g)
                bfs.append(bfs_list)
                indices.append(i)
                g_list.append(g)
            return list(zip(sub_g_list, bfs, indices, g_list))
        else:
            sample_index = int(len(bfs_list) * ratio)
            if sample_index < 2:
                sample_index = 2
            sub_g = get_subgraph(bfs_list, sample_index, event_event_dict, event_entity_dict, entity_entity_dict, event_type, entity_type)
            sub_g_list.append(sub_g)
            bfs.append(bfs_list)
            indices.append(sample_index)
            g_list.append(g)
            return list(zip(sub_g_list, bfs, indices, g_list))


class GraphDataset(Dataset):
    """
    Args:
        root: directory of raw txt files of a graph dataset
        mode: mode of dataset, mode_choices=['train', 'eval']
        sample_ratio: the ratio to determine the position of sample index in test graphs, in the range of [0,1). If it's
                        0, test graphs will include all sample possibilities.
    """
    def __init__(self, root, mode='train', sample_ratio=0., transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.mode = mode
        self.ratio = sample_ratio
        if self.ratio < 0 or self.ratio >= 1:
            raise Exception("invalid sample ratio, ratio should be in [0,1)")
        with open(self.root + '/global_types2id.txt', 'r') as f:
            self.num_node_types = int(f.readline())
            self.num_event_types = 0
            line = f.readline()
            while line:
                if 'Entities' not in line:
                    self.num_event_types += 1
                line = f.readline()
            self.num_entity_types = self.num_node_types - self.num_event_types
        with open(self.root + '/global_Relations2id.txt', 'r') as f:
            self.num_rels = int(f.readline())

    @property
    def raw_file_names(self):
        raw_path_list = []
        for _, dirs, _ in os.walk(self.raw_dir):
            for dir_name in dirs:
                raw_path_list.append(dir_name)
        return raw_path_list

    @property
    def processed_file_names(self):
        '''
        processed_file = []
        for i in range(len(self.raw_file_names)):
            processed_file.append('data_%d' % i)
        return processed_file
        '''
        return []

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        """
        Given an complex event idx, return its graph object "g", subgraph "sub_g" after sampling in its bfs_list,
        and the sampled node id "sampled_node"
        """
        raw_path = self.raw_file_names[idx]
        graph_path = os.path.join(self.raw_dir, raw_path) + '/'
        if self.mode == 'train':
            sub_g, bfs_list, sample_index, g = get_single_graph(graph_path, self.mode, self.ratio)
            return sub_g, bfs_list, sample_index, g
        elif self.mode == 'eval':
            test_graph_list = get_single_graph(graph_path, self.mode, self.ratio)
            return test_graph_list
        else:
            raise Exception("invalid mode of dataset, mode_choices=['train', 'eval']")
