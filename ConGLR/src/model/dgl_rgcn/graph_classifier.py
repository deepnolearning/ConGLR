from .gcn_model import UniGCN
from dgl import mean_nodes
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class GraphClassifier(nn.Module):
    def __init__(self, params, relation2id):  # in_dim, h_dim, rel_emb_dim, out_dim, num_rels, num_bases):
        super().__init__()

        self.params = params
        self.relation2id = relation2id
        self.relation_list = list(self.relation2id.values())
        self.no_jk = self.params.no_jk

        self.gnn = UniGCN(params)  # in_dim, h_dim, h_dim, num_rels, num_bases)


        num_final_gcn = self.params.num_gcn_layers
        if self.no_jk:  # 仅仅使用最后一层的输出
            num_final_gcn = 1

        if self.params.add_ht_emb:
            self.fc_layer = nn.Linear(6 * num_final_gcn * self.params.emb_dim, 1)
        else:
            self.fc_layer = nn.Linear(4 * num_final_gcn * self.params.emb_dim, 1)

    def forward(self, data):
        g, cg, rel_labels = data

        local_g = g.local_var()
        in_deg = local_g.in_degrees(range(local_g.number_of_nodes())).float().numpy()
        in_deg[in_deg == 0] = 1  # 最小化入度为1
        node_norm = 1.0 / in_deg
        local_g.ndata['norm'] = node_norm
        local_g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        norm = local_g.edata['norm']  # 添加边的norm

        local_cg = cg.local_var()
        in_deg = local_cg.in_degrees(range(local_cg.number_of_nodes())).float().numpy()
        in_deg[in_deg == 0] = 1  # 最小化入度为1
        node_norm = 1.0 / in_deg
        local_cg.ndata['norm'] = node_norm
        local_cg.apply_edges(lambda edges: {'norm': edges.dst['norm']})
        cg_norm = local_cg.edata['norm']  # 添加边的norm

        if self.params.gpu >= 0:
            norm = norm.cuda(device=self.params.gpu)
            cg_norm = cg_norm.cuda(device=self.params.gpu)

        # g.ndata['h'] = self.gnn(g, norm)
        target_rel_emd, path_agg_emd = self.gnn(g, norm, cg_norm)

        g_out = mean_nodes(g, 'repr')  # 根据特征repr 析出图整体表示
        cg_out = mean_nodes(cg, 'repr')

        head_ids = (g.ndata['id'] == 1).nonzero().squeeze(1)  # batch中 可能存在多个
        head_embs = g.ndata['repr'][head_ids]

        tail_ids = (g.ndata['id'] == 2).nonzero().squeeze(1)
        tail_embs = g.ndata['repr'][tail_ids]

        g_rep = torch.cat([g_out,  head_embs, tail_embs, cg_out, target_rel_emd, path_agg_emd], dim=1)

        output = self.fc_layer(g_rep)
        return output

