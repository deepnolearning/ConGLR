# -*- coding: utf-8 -*-
# 加载dataset相关
import lmdb
import logging
import struct
import dgl
import numpy as np
import torch
from torch.utils.data import Dataset

from process_utils import process_files
from save_utils import deserialize
from graph_utils import ssp_multigraph_to_dgl
from context_graph import get_context_graph


class SubgraphDataset(Dataset):
    """Extracted, labeled, subgraph dataset -- DGL Only"""

    def __init__(self, db_path, id2revid, db_name_pos, db_name_neg, raw_data_paths, included_relations=None,
                 add_traspose_rels=False, num_neg_samples_per_link=1, use_kge_embeddings=False, dataset='',
                 kge_model='', file_name=''):

        self.main_env = lmdb.open(db_path, readonly=True, max_dbs=3, lock=False)
        self.db_pos = self.main_env.open_db(db_name_pos.encode())
        ##### del neg
        self.db_neg = self.main_env.open_db(db_name_neg.encode())
        self.node_features, self.kge_entity2id = (None, None)
        self.num_neg_samples_per_link = num_neg_samples_per_link
        self.file_name = file_name
        self.relid2revid = id2revid

        ssp_graph, __, __, __, id2entity, id2relation = process_files(raw_data_paths, included_relations)
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
            cgraph_pos = get_context_graph(subgraph_pos)
        subgraphs_neg = []
        cgraphs_neg = []
        r_labels_neg = []
        g_labels_neg = []
        with self.main_env.begin(db=self.db_neg) as txn:
            for i in range(self.num_neg_samples_per_link):
                str_id = '{:08}'.format(index + i * (self.num_graphs_pos)).encode('ascii')
                nodes_neg, r_label_neg, g_label_neg, n_labels_neg = deserialize(txn.get(str_id)).values()
                graph_neg = self._prepare_subgraphs(nodes_neg, r_label_neg, n_labels_neg)
                subgraphs_neg.append(graph_neg)
                cgraphs_neg.append(get_context_graph(graph_neg))
                r_labels_neg.append(r_label_neg)
                g_labels_neg.append(g_label_neg)

        return subgraph_pos, cgraph_pos, g_label_pos, r_label_pos, subgraphs_neg, cgraphs_neg, g_labels_neg, r_labels_neg

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

            # 增加逆关系
            # rev_id = self.relid2revid[r_label]
            # edges_btw_roots = subgraph.edge_id(1, 0)  # 返回0 1 节点之间的边的id
            # rel_link = np.nonzero(subgraph.edata['type'][edges_btw_roots] == rev_id)
            # if rel_link.squeeze().nelement() == 0:  # 如果没有目标关系，进行添加
            #     subgraph.add_edge(1, 0)
            #     subgraph.edata['type'][-1] = torch.tensor(rev_id).type(torch.LongTensor)
            #     subgraph.edata['label'][-1] = torch.tensor(r_label).type(torch.LongTensor)

        # print(subgraph.edata['type'])
        # print(subgraph.edata['label'])
        # else:
        #     print('---'*100)

        # map the id read by GraIL to the entity IDs as registered by the KGE embeddings
        kge_nodes = [self.kge_entity2id[self.id2entity[n]] for n in nodes] if self.kge_entity2id else None
        n_feats = self.node_features[kge_nodes] if self.node_features is not None else None
        subgraph = self._prepare_features(subgraph, n_labels, n_feats)

        # 删除没有关系的节点（节点表示不会更新，影响整体图的表示）
        # in_degrees = subgraph.in_degrees()
        # out_degrees = subgraph.out_degrees()
        # degrees = in_degrees + out_degrees
        # subgraph.remove_nodes(torch.where(degrees==0)[0])

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