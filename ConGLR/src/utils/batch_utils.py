# -*- coding: utf-8 -*-
# batch操作相关
import torch
import dgl


def collate_dgl(samples):
    # The input `samples` is a list of pairs
    graphs_pos, cgraphs_pos, g_labels_pos, r_labels_pos, graphs_negs, cgraphs_negs, g_labels_negs, r_labels_negs = map(list, zip(*samples))
    batched_graph_pos = dgl.batch(graphs_pos)
    batched_cgraph_pos = dgl.batch(cgraphs_pos)

    graphs_neg = [item for sublist in graphs_negs for item in sublist]
    cgraphs_neg = [item for sublist in cgraphs_negs for item in sublist]
    g_labels_neg = [item for sublist in g_labels_negs for item in sublist]
    r_labels_neg = [item for sublist in r_labels_negs for item in sublist]

    batched_graph_neg = dgl.batch(graphs_neg)
    batched_cgraph_neg = dgl.batch(cgraphs_neg)

    # 修改entity pair中实体在batch_graph中的对应  # 对正样本的处理？
    # print('--' * 100)
    # print(batched_cgraph_pos.ndata['f3'])
    # print(batched_graph_pos.batch_num_nodes)

    batch_nodes = batched_graph_pos.batch_num_nodes
    batch_cg_nodes = batched_cgraph_pos.batch_num_nodes
    ent = batched_cgraph_pos.ndata.pop('f3')
    start = 0
    end = 0
    p = 0
    for i in range(len(batch_cg_nodes)):
        end = batch_cg_nodes[i] + start
        p_value = torch.zeros_like(ent[start:end])  # 偏移矩阵
        p_value[ent[start:end] >= 0] = p
        ent[start:end] += p_value
        start = end
        p += batch_nodes[i]
    batched_cgraph_pos.ndata['f3'] = ent

    # print(sum(batch_nodes))
    # print(torch.max(batched_cgraph_pos.ndata['f3']))
    # print(batched_cgraph_pos.ndata['f3'])
    #
    # print(batched_cgraph_neg.ndata['f3'])
    # print(batched_graph_neg.batch_num_nodes)
    batch_nodes = batched_graph_neg.batch_num_nodes
    batch_cg_nodes = batched_cgraph_neg.batch_num_nodes
    ent = batched_cgraph_neg.ndata.pop('f3')
    start = 0
    end = 0
    p = 0
    for i in range(len(batch_cg_nodes)):
        end = batch_cg_nodes[i] + start
        p_value = torch.zeros_like(ent[start:end])  # 偏移矩阵
        p_value[ent[start:end] >= 0] = p
        ent[start:end] += p_value
        start = end
        p += batch_nodes[i]
    batched_cgraph_neg.ndata['f3'] = ent

    # print('**'*10)
    # print(sum(batch_nodes))
    # print(torch.max(batched_cgraph_neg.ndata['f3']))
    # print(batched_cgraph_neg.ndata['f3'])

    return (batched_graph_pos, batched_cgraph_pos, r_labels_pos), g_labels_pos, (batched_graph_neg, batched_cgraph_neg, r_labels_neg), g_labels_neg


def send_graph_to_device(g, device):
    # nodes
    labels = g.node_attr_schemes()
    for l in labels.keys():
        g.ndata[l] = g.ndata.pop(l).to(device)

    # edges
    labels = g.edge_attr_schemes()
    for l in labels.keys():
        g.edata[l] = g.edata.pop(l).to(device)
    return g


def move_batch_to_device_dgl(batch, device):
    ((g_dgl_pos, cg_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, cg_dgl_neg, r_labels_neg), targets_neg) = batch

    targets_pos = torch.LongTensor(targets_pos).to(device=device)
    r_labels_pos = torch.LongTensor(r_labels_pos).to(device=device)

    targets_neg = torch.LongTensor(targets_neg).to(device=device)
    r_labels_neg = torch.LongTensor(r_labels_neg).to(device=device)

    g_dgl_pos = send_graph_to_device(g_dgl_pos, device)
    cg_dgl_pos = send_graph_to_device(cg_dgl_pos, device)
    g_dgl_neg = send_graph_to_device(g_dgl_neg, device)
    cg_dgl_neg = send_graph_to_device(cg_dgl_neg, device)

    return ((g_dgl_pos, cg_dgl_pos, r_labels_pos), targets_pos, (g_dgl_neg, cg_dgl_neg, r_labels_neg), targets_neg)