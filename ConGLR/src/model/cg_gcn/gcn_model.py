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

        # cg输入关系表示
        self.rel_emb = nn.Parameter(torch.Tensor(params.num_rels + 1, params.rel_emb_dim))  # 多一个关系向量用于表示空关系
        # torch.nn.init.normal_(self.rel_emb)
        nn.init.xavier_uniform_(self.rel_emb, gain=nn.init.calculate_gain('relu'))

        self.line1_ent = nn.Linear(self.inp_dim, self.emb_dim)  # 初始化实体嵌入转换
        self.line1_rel = nn.Linear(self.emb_dim*3, self.emb_dim)  # 关系转化
        # self.line1_edge = nn.Linear(self.emb_dim, self.emb_dim)  # 初始化edge

        # self.ent_emb = nn.Parameter(torch.zeros(1, self.inp_dim))  # 多一个实体向量用于表示空实体
        # torch.nn.init.normal_(self.ent_emb)
        # self.ent_emb_hidden = nn.Parameter(torch.zeros(self.num_hidden_layers, 1, self.emb_dim))  # 多一个实体向量用于表示空实体（在迭代GCN时）
        # torch.nn.init.normal_(self.ent_emb_hidden)

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

        # self.line_ent = nn.Linear(self.inp_dim, self.emb_dim)
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers_rgcn = nn.ModuleList()
        self.layers_cggcn = nn.ModuleList()

        # g/cg图处理方式
        self.layers_rgcn.append(self.build_input_layer())
        self.layers_cggcn.append(self.build_cg_input_layer())
        for idx in range(self.num_hidden_layers - 1):
            self.layers_rgcn.append(self.build_hidden_layer())
            self.layers_cggcn.append(self.build_cg_hidden_layer())

    def build_input_layer(self):  # device=self.device activation之前
        return RGCNLayer(self.emb_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.aug_num_rels, self.num_bases , device=self.device, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout, is_input_layer=True, has_attn=self.has_attn, no_jk=self.no_jk)

    def build_hidden_layer(self):
        return RGCNLayer(self.emb_dim, self.emb_dim, self.aggregator, self.attn_rel_emb_dim, self.aug_num_rels, self.num_bases, device=self.device, activation=F.relu,
                         dropout=self.dropout, edge_dropout=self.edge_dropout, has_attn=self.has_attn, no_jk=self.no_jk)


    def build_cg_input_layer(self):
        return CGGCN(self.emb_dim, self.emb_dim, self.cg_aggregator, self.aug_num_rels, self.device, activation=F.relu,
                     dropout=self.dropout, edge_dropout=self.edge_dropout, is_input_layer=True, no_jk=self.no_jk)

    def build_cg_hidden_layer(self):
        return CGGCN(self.emb_dim, self.emb_dim, self.cg_aggregator, self.aug_num_rels, self.device, activation=F.relu,
                     dropout=self.dropout, edge_dropout=self.edge_dropout, is_input_layer=False, no_jk=self.no_jk)

    # def forward(self, g, norm):
    #     for layer in self.layers:
    #         layer(g, self.attn_rel_emb, norm)
    #     return g.ndata.pop('h')

    # 初始化cg
    # 通过关系表示和g中实体表示获取cg中节点表示
    def get_first_cg_feat(self, g, cg, cg_agg='mean'):  # mean max

        index_offset = 1  # index偏移量
        f1 = cg.ndata['f1'] + index_offset
        f2 = cg.ndata['f2'] + index_offset
        f3 = cg.ndata['f3']
        index1 = cg.ndata['index1']
        index2 = cg.ndata['index2']
        index3 = cg.ndata['index3']

        # relation/path
        path_max_len = f1.size()[1]
        path_feat = torch.index_select(self.rel_emb, dim=0, index=f1.view(-1)).view(f3.size()[0], path_max_len, -1)
        rel_feat = torch.index_select(self.rel_emb, dim=0, index=f2)

        # entity
        g_feats = self.line1_ent(g.ndata['feat'])  # 中间层 表示h 初始化
        g.ndata['feat'] = g_feats  # 初始化就转换

        cg_feat = torch.zeros((f3.size(0), self.emb_dim)).to(device=self.device)
        cg_feat[f3 != -1] = g_feats[f3[f3 != -1]]
        ent_feat = cg_feat  # 初始化嵌入转换
        # ent_feat = cg_feat.view(-1, 2, self.emb_dim)  # N 2 F

        # if cg_agg == 'mean':
        #     path_feat = torch.mean(path_feat, dim=1)
        # elif cg_agg == 'max':
        #     path_feat = torch.max(path_feat, dim=1).values
        path_feat = self.line1_rel(path_feat.view(f3.size()[0], -1))

        cg_feat = path_feat * index1.unsqueeze(1) + rel_feat * index2.unsqueeze(1) + ent_feat * index3.unsqueeze(1)

        # cg_feat = torch.relu(cg_feat)
        return cg_feat


    # 仅从g中获取entity pair的信息到cg
    def get_cg_ent_feat(self, g, cg):  # mean max
        f3 = cg.ndata['f3']
        index3 = cg.ndata['index3']

        g_feats = g.ndata['feat']  # 中间层 表示h  feat具有残差连接的（调试哪个更好一点）
        cg_feat = torch.zeros((f3.size(0), self.emb_dim)).to(device=self.device)
        cg_feat[f3 != -1] = g_feats[f3[f3 != -1]]
        ent_feat = cg_feat.detach()  # 是否需要detach 不要梯度回传
        # ent_feat = cg_feat  # 是否需要detach 不要梯度回传

        cg_feat = ent_feat * index3.unsqueeze(1)

        return cg_feat


    def forward(self, g, norm, cg, cg_norm):
        # batch_rel_emds = self.rel_emb.repeat((self.batch_size, 1, 1))
        # expand 不分配新的空间
        # print(batch_rel_emds.size())


        target_rel = None
        path_agg = None
        for i in range(self.num_hidden_layers):
            rgcn_layer = self.layers_rgcn[i]
            cggcn_layer = self.layers_cggcn[i]

            # cg特征转换
            if i == 0:
                # 初始化cg节点的特征
                cg_feat = self.get_first_cg_feat(g, cg, cg_agg=self.cg_agg)
                cg.ndata['feat'] = cg_feat
            else:
                index3 = cg.ndata['index3']
                cg_feat = self.get_cg_ent_feat(g, cg)
                cg.ndata['feat'][index3==1] = cg.ndata['feat'][index3==1] * 0.5 + cg_feat[index3==1] * 0.5  # 中间特征用feat表示

            batch_rel_emds, target_rel_emd, path_agg_emd = cggcn_layer(cg, cg_norm)  # batch_rel_emds向g中传递cg的relation信息

            # 获取edge_rel_emd
            index_offset = 1
            batch_edges = g.batch_num_edges
            edge_types = g.edata['type'] + index_offset
            edge_rel_emd = []
            target_rel_emd_new = []
            index_start = 0
            index_end = 0
            # 头尾信息
            head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)  # batch中 可能存在多个
            head_embs = g.ndata['feat'][head_ids]  # repr
            tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
            tail_embs = g.ndata['feat'][tail_ids]
            head_embs_new = []
            tail_embs_new = []
            for kk, num_edges in enumerate(batch_edges):
                index_end = index_start + num_edges
                temp_edge_rel_emd = torch.index_select(batch_rel_emds[kk], dim=0,
                                                       index=edge_types[index_start:index_end])
                edge_rel_emd.append(temp_edge_rel_emd)

                temp_tar = [target_rel_emd[kk]] * num_edges
                target_rel_emd_new.extend(temp_tar)
                temp_tar = [head_embs[kk]] * num_edges
                head_embs_new.extend(temp_tar)
                temp_tar = [tail_embs[kk]] * num_edges
                tail_embs_new.extend(temp_tar)

                index_start = index_end

            edge_rel_emd = torch.cat(edge_rel_emd, dim=0).detach()
            target_rel_emd_new = torch.stack(target_rel_emd_new, dim=0).detach()

            # edge_rel_emd = torch.cat(edge_rel_emd, dim=0)
            # target_rel_emd_new = torch.stack(target_rel_emd_new, dim=0)

            rgcn_layer(g, norm, edge_rel_emd, target_rel_emd_new)  # h

            if i != 0 and self.no_jk == False:  # 多层结果拼接
                target_rel = torch.cat([target_rel, target_rel_emd], dim=1)
                path_agg = torch.cat([path_agg, path_agg_emd], dim=1)
            else:
                target_rel = target_rel_emd
                path_agg = path_agg_emd
            # cg中entity pair表示中加入g的节点信息


        '''
        # 关系、实体表示不进行交互
        for i in range(self.num_hidden_layers):
            cggcn_layer = self.layers_cggcn[i]
            # cg特征转换
            if i == 0:
                # 初始化cg节点的特征
                cg_feat = self.get_first_cg_feat(g, cg, cg_agg=self.cg_agg)
                cg.ndata['feat'] = cg_feat
            batch_rel_emds, target_rel_emd, path_agg_emd = cggcn_layer(cg, cg_norm)  # batch_rel_emds向g中传递cg的relation信息
            if i != 0 and self.no_jk == False:  # 多层结果拼接
                target_rel = torch.cat([target_rel, target_rel_emd], dim=1)
                path_agg = torch.cat([path_agg, path_agg_emd], dim=1)
            else:
                target_rel = target_rel_emd
                path_agg = path_agg_emd
            # cg中entity pair表示中加入g的节点信息

        for i in range(self.num_hidden_layers):
            rgcn_layer = self.layers_rgcn[i]
            rgcn_layer(g, norm, batch_rel_emds, target_rel_emd)  # h
        '''

        return target_rel, path_agg

