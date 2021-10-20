"""
File based off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import RGCNBasisLayer as RGCNLayer
from .layers import CGGCN

from .aggregators import SumAggregator, MLPAggregator, GRUAggregator, CGAggregator


class UniGCN(nn.Module):  # 处理两个图的gcn联合模型
    def __init__(self, params):
        super(UniGCN, self).__init__()

        # self.max_label_value = params.max_label_value
        self.inp_dim = params.inp_dim
        self.emb_dim = params.emb_dim
        self.attn_rel_emb_dim = params.attn_rel_emb_dim  # g图
        self.num_rels = params.num_rels
        self.aug_num_rels = params.aug_num_rels
        self.num_bases = params.num_bases
        self.num_hidden_layers = params.num_gcn_layers
        self.dropout = params.dropout
        self.edge_dropout = params.edge_dropout
        # self.aggregator_type = params.gnn_agg_type
        self.has_attn = params.has_attn  # g图
        self.cg_agg = params.cg_agg_type
        self.no_jk = params.no_jk

        self.device = params.device
        self.batch_size = params.batch_size

        self.rel_emb = nn.Parameter(torch.zeros(self.params.num_rels + 1, self.params.rel_emb_dim))  # 多一个关系向量用于表示空关系
        torch.nn.init.normal_(self.rel_emb)
        self.ent_emb = nn.Parameter(torch.zeros(1, self.inp_dim))  # 多一个实体向量用于表示空实体
        torch.nn.init.normal_(self.ent_emb)
        self.ent_emb_hidden = nn.Parameter(torch.zeros(self.num_hidden_layers - 1, 1, self.emb_dim))  # 多一个实体向量用于表示空实体（在迭代GCN时）
        torch.nn.init.normal_(self.ent_emb_hidden)

        # initialize aggregators for input and hidden layers
        if params.gnn_agg_type == "sum":
            self.aggregator = SumAggregator(self.emb_dim)
        elif params.gnn_agg_type == "mlp":
            self.aggregator = MLPAggregator(self.emb_dim)
        elif params.gnn_agg_type == "gru":
            self.aggregator = GRUAggregator(self.emb_dim)
        self.cg_aggregator = CGAggregator(self.emb_dim)

        # initialize basis weights for input and hidden layers
        # self.input_basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.emb_dim))
        # self.basis_weights = nn.Parameter(torch.Tensor(self.num_bases, self.emb_dim, self.emb_dim))

        # create rgcn layers
        self.build_model()

        # create initial features
        # self.features = self.create_features()

    # def create_features(self):
    #     features = torch.arange(self.inp_dim).to(device=self.device)
    #     return features

    def build_model(self):
        self.layers_rgcn = nn.ModuleList()
        self.layers_cggcn = nn.ModuleList()

        # g/cg图处理方式
        self.layers_rgcn.append(self.build_input_layer())
        self.layers_cggcn.append(self.build_cg_input_layer())
        for idx in range(self.num_hidden_layers - 1):
            self.layers_rgcn.append(self.build_hidden_layer())
            self.layers_cggcn.append(self.build_cg_hidden_layer())

    def build_input_layer(self):
        return RGCNLayer(self.inp_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.aug_num_rels, self.num_bases, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout, is_input_layer=True, has_attn=self.has_attn, no_jk=self.no_jk)

    def build_hidden_layer(self):
        return RGCNLayer(self.emb_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.aug_num_rels, self.num_bases, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout, has_attn=self.has_attn, no_jk=self.no_jk)


    def build_cg_input_layer(self):
        return CGGCN(self,self.inp_dim, self.emb_dim, self.cg_aggregator, self.aug_num_rels, self.device, activation=F.relu,
                     dropout=self.dropout, is_input_layer=True, no_jk=self.no_jk)

    def build_cg_hidden_layer(self):
        return CGGCN(self,self.emb_dim, self.emb_dim, self.cg_aggregator, self.aug_num_rels, self.device, activation=F.relu,
                     dropout=self.dropout, is_input_layer=True, no_jk=self.no_jk)

    # def forward(self, g, norm):
    #     for layer in self.layers:
    #         layer(g, self.attn_rel_emb, norm)
    #     return g.ndata.pop('h')

    def forward(self, g, cg, norm):
        batch_rel_emds = self.rel_emb.repeat((self.batch_size, 1, 1))

        # 通过关系表示和g中实体表示获取cg中节点表示
        def get_cg_feat(g, cg, empty_ent_emd, cg_agg='mean'):  # mean max
            index_offset = 1  # index偏移量
            f1 = cg.ndata['f1'] + index_offset
            f2 = cg.ndata['f2'] + index_offset
            f3 = cg.ndata['f3'] + index_offset
            index1 = cg.ndata['index1']
            index2 = cg.ndata['index2']
            index3 = cg.ndata['index3']

            # relation/path
            path_max_len = f1.size()[1]
            cg_batch_nodes = cg.batch_num_nodes  # batch中每个子图的节点列表
            cg_index_start = 0
            cg_index_end = 0
            path_feat = None
            rel_feat = None
            for i in range(self.batch_size):
                cg_index_end = cg_index_start + cg_batch_nodes[i]
                temp_path_feat = torch.index_select(batch_rel_emds[i], dim=0, index=f1[cg_index_start:cg_index_end].view(-1))\
                    .view(-1, path_max_len)  # N, P, F
                temp_rel_feat = torch.index_select(batch_rel_emds[i], dim=0, index=f2[cg_index_start:cg_index_end])  # N, F
                if i == 0:
                    path_feat = temp_path_feat
                    rel_feat = temp_rel_feat
                else:
                    path_feat = torch.cat((path_feat, temp_path_feat), dim=0)
                    rel_feat = torch.cat((rel_feat, temp_rel_feat), dim=0)
                cg_index_start = cg_index_end

            # entity pair
            g_feats = g.ndata['feat']  # N, F
            g_feats = torch.cat((empty_ent_emd, g_feats), dim=0)
            pair_feat = torch.index_select(g_feats, dim=0, index=f3.view(-1)).view(-1, 2)  # N, 2, F

            if cg_agg == 'mean':
                path_feat = torch.mean(path_feat, dim=1)
                pair_feat = torch.mean(pair_feat, dim=1)
            elif cg_agg == 'max':
                path_feat = torch.max(path_feat, dim=1).values
                pair_feat = torch.max(pair_feat, dim=1).values
            cg_feat = path_feat.T * index1 + rel_feat.T * index2 + pair_feat.T * index3
            cg_feat = cg_feat.T
            # cg.ndata['feat'] = cg_feat.to(device=self.device)
            return cg_feat

        # 仅从g中获取entity pair的信息到cg
        def get_cg_pair_feat(g, cg, empty_ent_emd, cg_agg='mean'):  # mean max
            index_offset = 1  # index偏移量
            f3 = cg.ndata['f3'] + index_offset
            index3 = cg.ndata['index3']

            # entity pair
            g_feats = g.ndata['h']  # 中间层 表示h
            g_feats = torch.cat((empty_ent_emd, g_feats), dim=0)
            pair_feat = torch.index_select(g_feats, dim=0, index=f3.view(-1)).view(-1, 2)  # N, 2, F

            if cg_agg == 'mean':
                pair_feat = torch.mean(pair_feat, dim=1)
            elif cg_agg == 'max':
                pair_feat = torch.max(pair_feat, dim=1).values
            cg_feat = pair_feat.T * index3
            cg_feat = cg_feat.T
            # cg.ndata['feat'] = cg_feat.to(device=self.device)
            return cg_feat

        # 初始化cg节点的特征
        cg_feat = get_cg_feat(g, cg, self.ent_emb, cg_agg=self.cg_agg)
        # cg.ndata['feat'] = cg_feat.to(device=self.device)
        cg.ndata['feat'] = cg_feat

        target_rel = None
        path_agg = None
        for i in range(self.num_hidden_layers):
            rgcn_layer = self.layers_rgcn[i]
            cggcn_layer = self.layers_cggcn[i]
            rgcn_layer(g, norm, batch_rel_emds)  # h
            batch_rel_emds, target_rel_emd, path_agg_emd = cggcn_layer(cg, batch_rel_emds)  # batch_rel_emds向g中传递cg的relation信息
            if i != 0 and self.no_jk == False:  # 多层结果拼接
                target_rel = torch.cat([target_rel, target_rel_emd], dim=1)
                path_agg = torch.cat([path_agg, path_agg_emd], dim=1)
            else:
                target_rel = target_rel_emd
                path_agg = path_agg_emd
            # cg中entity pair表示中加入g的节点信息
            if i != self.num_hidden_layers - 1:   # 最后一层不添加
                cg_feat = get_cg_pair_feat(g, cg, self.ent_emb_hidden[i], cg_agg=self.cg_agg)
                # cg.ndata['feat'] = cg.ndata['feat'] + cg_feat.to(device=self.device)
                cg.ndata['feat'] = cg.ndata['feat'] + cg_feat  # 中间特征用feat表示


        return target_rel, path_agg

