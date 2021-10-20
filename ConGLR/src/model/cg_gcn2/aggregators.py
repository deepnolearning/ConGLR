import abc
import torch.nn as nn
import torch
import torch.nn.functional as F


class Aggregator(nn.Module):
    def __init__(self, emb_dim):
        super(Aggregator, self).__init__()

    def forward(self, node):
        # node.mailbox里面的数据数据是对于目标尾节点的 [N, E, D] 节点数量、对应的边数量、信息表示
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (B, F)
        # nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (B, F)
        alpha = node.mailbox['alpha'].transpose(1, 2)
        nei_msg = torch.bmm(torch.softmax(alpha, dim=-1), node.mailbox['msg']).squeeze(1)  # (B, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    @abc.abstractmethod
    def update_embedding(curr_emb, nei_msg):
        raise NotImplementedError


class SumAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(SumAggregator, self).__init__(emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = nei_msg + curr_emb

        return new_emb


class MLPAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(MLPAggregator, self).__init__(emb_dim)
        self.linear = nn.Linear(2 * emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        inp = torch.cat((nei_msg, curr_emb), 1)
        new_emb = F.relu(self.linear(inp))

        return new_emb


class GRUAggregator(Aggregator):
    def __init__(self, emb_dim):
        super(GRUAggregator, self).__init__(emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def update_embedding(self, curr_emb, nei_msg):
        new_emb = self.gru(nei_msg, curr_emb)

        return new_emb


class CGAggregator(nn.Module):  # context graph 聚合器
    def __init__(self, emb_dim):
        super(CGAggregator, self).__init__()
        # self.linear = nn.Linear(2 * emb_dim, emb_dim)



    def forward(self, node):
        # node.mailbox里面的数据数据是对于目标尾节点的 [N, E, D] 节点数量、对应的边数量、信息表示
        curr_emb = node.mailbox['curr_emb'][:, 0, :]  # (N, F)
        nei_msg = torch.bmm(node.mailbox['alpha'].transpose(1, 2), node.mailbox['msg']).squeeze(1)  # (N, F)

        new_emb = self.update_embedding(curr_emb, nei_msg)

        return {'h': new_emb}

    def update_embedding(self, curr_emb, nei_msg):
        # inp = torch.cat((nei_msg, curr_emb), 1)
        # new_emb = F.relu(self.linear(inp))
        # new_emb = self.linear(inp)

        new_emb = nei_msg + curr_emb

        return new_emb
