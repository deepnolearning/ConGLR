"""
File baseed off of dgl tutorial on RGCN
Source: https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


class RGCNLayer(nn.Module):
    def __init__(self, inp_dim, out_dim, aggregator, bias=None, activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, no_jk=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.no_jk = no_jk

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        self.aggregator = aggregator

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if edge_dropout:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = Identity()

    # define how propagation is done in subclass
    def propagate(self, g, norm=None, batch_rel_emds=None):
        raise NotImplementedError

    def forward(self, g, norm=None, batch_rel_emds=None):

        self.propagate(g, norm, batch_rel_emds)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)

        g.ndata['h'] = node_repr

        if self.is_input_layer or self.no_jk:
            g.ndata['repr'] = g.ndata['h'].unsqueeze(1)
        else:  # 隐藏层  与上一层表示进行拼接  repr 后面都没有使用？
            g.ndata['repr'] = torch.cat([g.ndata['repr'], g.ndata['h'].unsqueeze(1)], dim=1)


class RGCNBasisLayer(RGCNLayer):
    def __init__(self, inp_dim, out_dim, aggregator, attn_rel_emb_dim, num_rels, num_bases=-1, bias=None,
                 activation=None, dropout=0.0, edge_dropout=0.0, is_input_layer=False, has_attn=False, no_jk=False):
        super(
            RGCNBasisLayer,
            self).__init__(
            inp_dim,
            out_dim,
            aggregator,
            bias,
            activation,
            dropout=dropout,
            edge_dropout=edge_dropout,
            is_input_layer=is_input_layer,
            no_jk=no_jk)
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.attn_rel_emb_dim = attn_rel_emb_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.is_input_layer = is_input_layer
        self.has_attn = has_attn

        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # add basis weights
        # self.weight = basis_weights
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.inp_dim, self.out_dim))
        self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        if self.has_attn:
            self.A1 = nn.Linear(2 * self.inp_dim + self.attn_rel_emb_dim, inp_dim)  # att1
            self.B1 = nn.Linear(inp_dim, 1)
            self.A2 = nn.Linear(2 * self.attn_rel_emb_dim, inp_dim)  # att2
            self.B2 = nn.Linear(inp_dim, 1)

        self.self_loop_weight = nn.Parameter(torch.Tensor(self.inp_dim, self.out_dim))

        nn.init.xavier_uniform_(self.self_loop_weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.w_comp, gain=nn.init.calculate_gain('relu'))

    def propagate(self, g, norm=None, batch_rel_emds=None):
        # generate all weights from bases
        if norm is not None:
            g.edata['norm'] = norm

        weight = self.weight.view(self.num_bases, self.inp_dim * self.out_dim)
        weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.inp_dim, self.out_dim)

        g.edata['w'] = self.edge_dropout(torch.ones(g.number_of_edges(), 1).to(weight.device))  # 随机删除一部分edge

        input_ = 'feat' if self.is_input_layer else 'h'

        # 提取关系表示
        batch_edges = g.batch_num_edges
        edge_types = g.edata['type']
        edge_labels = g.edata['label']
        edge_rel_emd = None  # E, D
        target_rel_emd = None  # E, D  edge关系表示+目标关系表示 用以计算注意力2
        index_start = 0
        index_end = 0
        for i, num_edges in enumerate(batch_edges):
            index_end = index_start + num_edges
            temp_edge_rel_emd = torch.index_select(batch_rel_emds[i], dim=0, index=edge_types[index_start:index_end])
            temp_target_rel_emd = torch.index_select(batch_rel_emds[i], dim=0, index=edge_labels[index_start])  # 只取一个就行 1 * F
            temp_target_rel_emd = temp_edge_rel_emd + temp_target_rel_emd
            if i == 0:
                edge_rel_emd = temp_edge_rel_emd
                target_rel_emd = temp_target_rel_emd
            else:
                edge_rel_emd = torch.cat((edge_rel_emd, temp_edge_rel_emd), dim=0)
                target_rel_emd = torch.cat((target_rel_emd, temp_target_rel_emd), dim=0)
            index_start = index_end



        def msg_func(edges):
            w = weight.index_select(0, edges.data['type'])  # [E, I, O]
            msg = edges.data['w'] * torch.bmm(edges.src[input_].unsqueeze(1), w).squeeze(1)    # [E, O]
            curr_emb = torch.mm(edges.dst[input_], self.self_loop_weight)  # (E, O)

            if self.has_attn:
                e1 = torch.cat([edges.src[input_], edges.dst[input_], edge_rel_emd], dim=1)
                a1 = torch.sigmoid(self.B1(torch.relu(self.A1(e1))))
                e2 = torch.cat([edge_rel_emd, target_rel_emd], dim=1)
                a2= torch.sigmoid(self.B2(torch.relu(self.A2(e2))))
                a = a1 + a2  # a1 * a2
            else:
                a = torch.ones((len(edges), 1)).to(device=w.device)

            if 'norm' in edges.data:
                msg = msg * edges.data['norm'].unsqueeze(1)
            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        g.update_all(msg_func, self.aggregator, None)


class CGGCN(nn.Module):  # context graph处理
    def __init__(self,inp_dim, out_dim, aggregator, num_rels, device, bias=None, activation=None, dropout=0.0, is_input_layer=False, no_jk=False):
        super(CGGCN, self).__init__()
        self.aggregator = aggregator
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.is_input_layer = is_input_layer
        self.no_jk = no_jk
        self.num_rels = num_rels
        self.device = device
        self.bias = bias
        self.activation = activation

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
            nn.init.xavier_uniform_(self.bias, gain=nn.init.calculate_gain('relu'))
        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self.W = nn.Parameter(torch.Tensor(2, inp_dim, out_dim))  # 关系转换矩阵
        self.self_loop_weight = nn.Parameter(torch.Tensor(inp_dim, out_dim))


    def forward(self, cg, batch_rel_emds):

        def msg_func(edges):
            w = self.W.index_select(0, edges.data['type'])  # [E, I, O]
            msg = torch.bmm(edges.src['feat'].unsqueeze(1), w).squeeze(1)  # [E, O]
            curr_emb = torch.mm(edges.dst['feat'], self.self_loop_weight)  # (E, O)

            a = torch.ones((len(edges), 1)).to(device=w.device)  # 后续增加注意力计算

            return {'curr_emb': curr_emb, 'msg': msg, 'alpha': a}

        cg.update_all(msg_func, self.aggregator, None)

        # 更新关系表示用于g的表示
        batch_nodes = cg.batch_num_nodes
        index1 = cg.ndata['index1']  # path 表示
        index2 = cg.ndata['index2']  # relation
        index_tar_rel = cg.ndata['rel_label']  # 目标关系
        index_offset = 1

        cg_node_feats = cg.ndata['h']
        # cg_path_feats = (cg_node_feats.T * index1).T
        # cg_rel_feats = (cg_node_feats.T * index2).T

        index_start = 0
        index_end = 0
        path_agg_emd = None
        target_rel_emd = None
        for i, nodes in enumerate(batch_nodes):
            index_end = index_start + nodes

            # relation
            new_rel_feats = torch.zeros(self.num_rels + 1, self.num_rels + 1).to(device=self.device)  # 额外一个表示空关系
            # temp_features = cg_rel_feats[index_start:index_end]
            node_features = cg_node_feats[index_start:index_end]
            temp_rels = index2[index_start:index_end] + index_offset
            new_rel_feats[temp_rels[temp_rels != 0]] = node_features[temp_rels != 0]  # 新关系表示
            new_rel_emds = batch_rel_emds[i] + new_rel_feats

            # 目标关系
            target_relation = index_tar_rel[index_start] + index_offset
            target_relation = new_rel_emds[target_relation]  # 1 * F

            # path表示
            path_index = index1[index_start:index_end]
            if len(path_index[path_index == 1]) == 0:  # 子图中没有path
                path_emd = torch.zeros(1, self.out_dim).to(device=self.device)
            else:  # path 表示与target relation 聚合
                temp_paths = index1[index_start:index_end]
                path_emd = node_features[temp_paths != 0]  # N * F

                alpha = torch.mm(path_emd, target_relation.T)  # N * 1
                path_emd = path_emd * alpha  # N * F
                path_emd = path_emd.mean(dim=0).unsqueeze(0)  # 1 * F

            if i == 0:
                target_rel_emd = target_relation
                path_agg_emd = path_emd
            else:
                target_rel_emd = torch.cat((target_rel_emd, target_relation), dim=0)
                path_agg_emd = torch.cat((path_agg_emd, path_emd), dim=0)

            batch_rel_emds[i] = new_rel_emds
            index_start = index_end


        node_repr = cg.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout:
            node_repr = self.dropout(node_repr)
        g.ndata['feat'] = node_repr

        if self.is_input_layer or self.no_jk:
            cg.ndata['repr'] = cg.ndata['feat'].unsqueeze(1)  # .unsqueeze(1)  应该去掉
        else:
            cg.ndata['repr'] = torch.cat([cg.ndata['repr'], cg.ndata['feat'].unsqueeze(1)], dim=1)

        return batch_rel_emds, target_rel_emd, path_agg_emd

