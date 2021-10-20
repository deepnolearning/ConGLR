# -*- coding: utf-8 -*-
# 构建子图的Context Graph
import lmdb
import logging
import struct
from collections import defaultdict
import numpy as np
import networkx as nx
import torch
import dgl
from torch.utils.data import Dataset

from process_utils import process_files
from save_utils import deserialize
from graph_utils import ssp_multigraph_to_dgl

class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='',
                 kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        ##### del neg
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
        self.id2relation = id2relation
        self.relation_list = list(id2relation.keys())
        self.num_rels = len(ssp_graph)

        # Add transpose matrices to handle both directions of relations.
        if add_traspose_rels:  # 增加逆关系，只需增加其对应的邻接矩阵就可以实现
            ssp_graph_t = [adj.T for adj in ssp_graph]
            ssp_graph += ssp_graph_t

        # the effective number of relations after adding symmetric adjacency matrices and/or self connections
        self.aug_num_rels = len(ssp_graph)

        self.graph = ssp_multigraph_to_dgl(ssp_graph)  # 生成一个大图
        self.ssp_graph = ssp_graph  # 邻接矩阵列表
        self.id2entity = id2entity
        self.id2relation = id2relation

        self.max_n_label = np.array([0, 0])
        with self.main_env.begin() as txn:
            self.max_n_label[0] = int.from_bytes(txn.get('max_n_label_sub'.encode()), byteorder='little')
            self.max_n_label[1] = int.from_bytes(txn.get('max_n_label_obj'.encode()), byteorder='little')

            self.avg_subgraph_size = struct.unpack('f', txn.get('avg_subgraph_size'.encode()))
            self.min_subgraph_size = struct.unpack('f', txn.get('min_subgraph_size'.encode()))
            self.max_subgraph_size = struct.unpack('f', txn.get('max_subgraph_size'.encode()))
            self.std_subgraph_size = struct.unpack('f', txn.get('std_subgraph_size'.encode()))

            self.avg_enc_ratio = struct.unpack('f', txn.get('avg_enc_ratio'.encode()))
            self.min_enc_ratio = struct.unpack('f', txn.get('min_enc_ratio'.encode()))
            self.max_enc_ratio = struct.unpack('f', txn.get('max_enc_ratio'.encode()))
            self.std_enc_ratio = struct.unpack('f', txn.get('std_enc_ratio'.encode()))

            self.avg_num_pruned_nodes = struct.unpack('f', txn.get('avg_num_pruned_nodes'.encode()))
            self.min_num_pruned_nodes = struct.unpack('f', txn.get('min_num_pruned_nodes'.encode()))
            self.max_num_pruned_nodes = struct.unpack('f', txn.get('max_num_pruned_nodes'.encode()))
            self.std_num_pruned_nodes = struct.unpack('f', txn.get('std_num_pruned_nodes'.encode()))

        logging.info(f"Max distance from sub : {self.max_n_label[0]}, Max distance from obj : {self.max_n_label[1]}")

        with self.main_env.begin(db=self.db_pos) as txn:
            self.num_graphs_pos = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')
        ########## del neg
        with self.main_env.begin(db=self.db_neg) as txn:
            self.num_graphs_neg = int.from_bytes(txn.get('num_graphs'.encode()), byteorder='little')

        self.__getitem__(0)  # 初始化 self.n_feat_dim

    def __getitem__(self, index):
        with self.main_env.begin(db=self.db_pos) as txn:  # positive 三元组
            str_id = '{:08}'.format(index).encode('ascii')
            nodes_pos, r_label_pos, g_label_pos, n_labels_pos = deserialize(txn.get(str_id)).values()  # 取值
            subgraph_pos = self._prepare_subgraphs(nodes_pos, r_label_pos, n_labels_pos)
        subgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                graph_neg = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                subgraphs_neg.append(graph_neg)
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, g_labels_neg, r_labels_neg

    def __len__(self):
        # return int(self.num_graphs_pos / 100)
        return self.num_graphs_pos

    def _prepare_subgraphs(self, nodes, r_label, n_labels):
        subgraph = dgl.DGLGraph(self.graph.subgraph(nodes))  # subgraph之后node重新编号
        subgraph.edata['type'] = self.graph.edata['type'][self.graph.subgraph(nodes).parent_eid]  # parent_eid: 完整图的edge id
        subgraph.edata['label'] = torch.tensor(r_label * np.ones(subgraph.edata['type'].shape), dtype=torch.long)  # label

        # print('**'*50)
        # print(subgraph.edata['type'])
        # print(subgraph.edata['label'])

        edges_btw_roots = subgraph.edge_id(0, 1)  # 返回0 1 节点之间的边的id
        rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == r_label)
        if rel_link.squeeze().nelement() == 0:  # 如果没有目标关系，进行添加
            # print('***'*100)
            subgraph.add_edge(0, 1)
            subgraph.edata['type'][-1] = torch.tensor(r_label).type(torch.LongTensor)
            subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features(subgraph, n_labels, n_feats)

        return subgraph

    def _prepare_features(self, subgraph, n_labels, n_feats=None):
        # One hot encode the node label feature and concat to n_featsure
        n_nodes = subgraph.number_of_nodes()
        label_feats = np.zeros((n_nodes, self.max_n_label[0] + 1 + self.max_n_label[1] + 1))
        label_feats[np.arange(n_nodes), n_labels[:, 0]] = 1
        label_feats[np.arange(n_nodes), self.max_n_label[0] + 1 + n_labels[:, 1]] = 1  # 双半径最短距离的one-hot编码
        n_feats = np.concatenate((label_feats, n_feats), axis=1) if n_feats is not None else label_feats
        subgraph.ndata['feat'] = torch.FloatTensor(n_feats)

        head_id = np.argwhere([label[0] == 0 and label[1] == 1 for label in n_labels])
        tail_id = np.argwhere([label[0] == 1 and label[1] == 0 for label in n_labels])
        n_ids = np.zeros(n_nodes)
        n_ids[head_id] = 1  # head
        n_ids[tail_id] = 2  # tail
        subgraph.ndata['id'] = torch.FloatTensor(n_ids)  # id 这个特征从1开始编号  只是作为标记

        self.n_feat_dim = n_feats.shape[1]  # Find cleaner way to do this -- i.e. set the n_feat_dim
        return subgraph


def get_context_graph(subgraph, max_paths = 200, max_path_len = 3, add_reverse=True):  # 4 是否增加逆关系 5 -> 10
    nodes = np.array(subgraph.nodes())
    edges = subgraph.edges()
    # print(nodes)
    # print(edges)
    edge_type = subgraph.edata['type']
    target_label = subgraph.edata['label'][0]

    type_edges = list(zip(edges[0], edges[1], edge_type))
    type_edges = np.array(type_edges)
    type_edges = [tuple(item) for item in type_edges]

    temp_nx = nx.MultiDiGraph(type_edges)
    path_list = []
    path_num = 0

    # print('关系路径:')
    for path in nx.all_simple_edge_paths(temp_nx, source=0, target=1, cutoff=max_path_len):
        path_num += 1
        if len(path) == 1 and path[0][-1] == target_label:  # 是待预测路径
            pass
        elif len(path) <= max_path_len:
            path_list.append(tuple([item[-1] for item in path]))
        if path_num > max_paths:  # 防止死循环
            break
    # for path in path_list:
    #     print(path)

    # print('mmmm',len(path_list))
    path_list = list(set(path_list))
    path_len = len(path_list)
    relations = list(set(np.array(edge_type)))
    relation_len = len(relations)
    entity_list = list(nodes)
    nodes_len = path_len + relation_len + len(entity_list)

    cg_nx = nx.MultiDiGraph()
    cg_nx.add_nodes_from(list(range(nodes_len)))
    rel_index_start = path_len
    ent_index_start = path_len + relation_len
    # Add edges
    nx_triplets = []
    for i, path in enumerate(path_list):
        for kk, rel in enumerate(path):  # 0 1 2
            rel_index = relations.index(rel) + rel_index_start
            nx_triplets.append((i, rel_index, {'type': kk}))  # path -> relation
    for h,t,r in type_edges:
        rel_index = relations.index(r) + rel_index_start
        h = h + ent_index_start
        t = t + ent_index_start
        nx_triplets.append((h, rel_index, {'type': max_path_len}))  # relation context
        nx_triplets.append((t, rel_index, {'type': max_path_len+1}))  # relation context
    sig_rel_num = max_path_len + 2
    if add_reverse:  # 增加逆关系
        for i, path in enumerate(path_list):
            for kk, rel in enumerate(path):  # 0 1 2
                rel_index = relations.index(rel) + rel_index_start
                nx_triplets.append((rel_index, i, {'type': kk + sig_rel_num}))  # path -> relation
        for h, t, r in type_edges:
            rel_index = relations.index(r) + rel_index_start
            h = h + ent_index_start
            t = t + ent_index_start
            nx_triplets.append((rel_index, h, {'type': max_path_len + sig_rel_num}))  # relation context
            nx_triplets.append((rel_index, t, {'type': max_path_len + 1 + sig_rel_num}))  # relation context

    cg_nx.add_edges_from(nx_triplets)

    cg_dgl = dgl.DGLGraph(multigraph=True)
    cg_dgl.from_networkx(cg_nx, edge_attrs=['type'])


    # 构造特征  缺失部分使用-1代替
    f1 = np.ones((nodes_len, max_path_len), dtype=np.int) * -1  # path特征
    f2 = np.ones(nodes_len, dtype=np.int) * -1  # relation特征
    f3 = np.ones(nodes_len, dtype=np.int) * -1  # ent特征
    tar_rel = np.zeros(nodes_len, dtype=np.int)  # 是否是目标关系

    for i, path in enumerate(path_list):
        f1[i][:len(path)] = np.array(path)
    for i, rel in enumerate(relations):
        f2[rel_index_start + i] = rel
        if rel == target_label:  # 目标关系index
            tar_rel[rel_index_start + i] = 1
    for i, ent in enumerate(entity_list):
        f3[ent_index_start + i] = ent

    index1 = np.zeros(nodes_len, dtype=np.int)
    index2 = np.zeros(nodes_len, dtype=np.int)
    index3 = np.zeros(nodes_len, dtype=np.int)
    index1[:rel_index_start] = 1
    index2[rel_index_start:ent_index_start] = 1
    index3[ent_index_start:] = 1

    cg_dgl.ndata['f1'] = f1
    cg_dgl.ndata['f2'] = f2
    cg_dgl.ndata['f3'] = f3
    cg_dgl.ndata['index1'] = index1
    cg_dgl.ndata['index2'] = index2
    cg_dgl.ndata['index3'] = index3
    cg_dgl.ndata['tar_rel'] = tar_rel  # 是否是目标关系
    cg_dgl.ndata['rel_label'] = np.array([target_label]*nodes_len)  # 目标关系

    return cg_dgl

def get_context_graph2(subgraph, id2relation, max_paths = 200, max_path_len = 3, add_reverse=True):  # 临时，用于展示
    nodes = np.array(subgraph.nodes())
    edges = subgraph.edges()
    # print(nodes)
    # print(edges)
    edge_type = subgraph.edata['type']
    target_label = subgraph.edata['label'][0]

    type_edges = list(zip(edges[0], edges[1], edge_type))
    type_edges = np.array(type_edges)
    type_edges = [tuple(item) for item in type_edges]

    temp_nx = nx.MultiDiGraph(type_edges)
    path_list = []
    path_num = 0

    print('--'*50)
    print('目标关系:')
    print(id2relation[target_label.item()])
    print('关系路径:')
    for path in nx.all_simple_edge_paths(temp_nx, source=0, target=1, cutoff=max_path_len):
        path_num += 1
        if len(path) == 1 and path[0][-1] == target_label:  # 是待预测路径
            pass
        elif len(path) <= max_path_len:
            path_list.append(tuple([item[-1] for item in path]))
        if path_num > max_paths:  # 防止死循环
            break

    # print('mmmm',len(path_list))
    path_list = list(set(path_list))
    for path in path_list:
        print([id2relation[item] for item in path])
    path_len = len(path_list)
    relations = list(set(np.array(edge_type)))
    relation_len = len(relations)
    entity_list = list(nodes)
    nodes_len = path_len + relation_len + len(entity_list)

    cg_nx = nx.MultiDiGraph()
    cg_nx.add_nodes_from(list(range(nodes_len)))
    rel_index_start = path_len
    ent_index_start = path_len + relation_len
    # Add edges
    nx_triplets = []
    for i, path in enumerate(path_list):
        for kk, rel in enumerate(path):  # 0 1 2
            rel_index = relations.index(rel) + rel_index_start
            nx_triplets.append((i, rel_index, {'type': kk}))  # path -> relation
    for h,t,r in type_edges:
        rel_index = relations.index(r) + rel_index_start
        h = h + ent_index_start
        t = t + ent_index_start
        nx_triplets.append((h, rel_index, {'type': max_path_len}))  # relation context
        nx_triplets.append((t, rel_index, {'type': max_path_len+1}))  # relation context
    sig_rel_num = max_path_len + 2
    if add_reverse:  # 增加逆关系
        for i, path in enumerate(path_list):
            for kk, rel in enumerate(path):  # 0 1 2
                rel_index = relations.index(rel) + rel_index_start
                nx_triplets.append((rel_index, i, {'type': kk + sig_rel_num}))  # path -> relation
        for h, t, r in type_edges:
            rel_index = relations.index(r) + rel_index_start
            h = h + ent_index_start
            t = t + ent_index_start
            nx_triplets.append((rel_index, h, {'type': max_path_len + sig_rel_num}))  # relation context
            nx_triplets.append((rel_index, t, {'type': max_path_len + 1 + sig_rel_num}))  # relation context

    cg_nx.add_edges_from(nx_triplets)

    cg_dgl = dgl.DGLGraph(multigraph=True)
    cg_dgl.from_networkx(cg_nx, edge_attrs=['type'])


    # 构造特征  缺失部分使用-1代替
    f1 = np.ones((nodes_len, max_path_len), dtype=np.int) * -1  # path特征
    f2 = np.ones(nodes_len, dtype=np.int) * -1  # relation特征
    f3 = np.ones(nodes_len, dtype=np.int) * -1  # ent特征
    tar_rel = np.zeros(nodes_len, dtype=np.int)  # 是否是目标关系

    for i, path in enumerate(path_list):
        f1[i][:len(path)] = np.array(path)
    for i, rel in enumerate(relations):
        f2[rel_index_start + i] = rel
        if rel == target_label:  # 目标关系index
            tar_rel[rel_index_start + i] = 1
    for i, ent in enumerate(entity_list):
        f3[ent_index_start + i] = ent

    index1 = np.zeros(nodes_len, dtype=np.int)
    index2 = np.zeros(nodes_len, dtype=np.int)
    index3 = np.zeros(nodes_len, dtype=np.int)
    index1[:rel_index_start] = 1
    index2[rel_index_start:ent_index_start] = 1
    index3[ent_index_start:] = 1

    cg_dgl.ndata['f1'] = f1
    cg_dgl.ndata['f2'] = f2
    cg_dgl.ndata['f3'] = f3
    cg_dgl.ndata['index1'] = index1
    cg_dgl.ndata['index2'] = index2
    cg_dgl.ndata['index3'] = index3
    cg_dgl.ndata['tar_rel'] = tar_rel  # 是否是目标关系
    cg_dgl.ndata['rel_label'] = np.array([target_label]*nodes_len)  # 目标关系

    return cg_dgl

def view_dataset():  # 原始dataset
    data_path = '/data/linqika/2021project/InductiveKG/data/nell_v2/subgraphs_en_True_neg_1_hop_3/'
    file_paths = {
        'train': '/data/linqika/2021project/InductiveKG/data/nell_v2/train.txt',
        'valid': '/data/linqika/2021project/InductiveKG/data/nell_v2/valid.txt',
    }
    train_dataset = SubgraphDataset(data_path, 'train_pos', 'train_neg', file_paths)
    valid_dataset = SubgraphDataset(data_path, 'valid_pos', 'valid_neg', file_paths)

    for i, item in enumerate(valid_dataset):  # 2 12 12, 3 23 53, 4 19 32
        # a = item[0]
        # print(i, item[0].number_of_nodes(), item[0].number_of_edges())
        # break
        if i != 70:
            temp_dgl = item[0]
            cg_dgl = get_context_graph2(temp_dgl,valid_dataset.id2relation)
            # print(cg_dgl.ndata)
            # print(cg_dgl.edata)
            # print(cg_dgl.edges())
            # break

view_dataset()
