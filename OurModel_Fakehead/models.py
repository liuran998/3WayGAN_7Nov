import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from args import args
from utils import uniform
import numpy as np

#New Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


def score(embedding, rel_embed, src, rel, dst):
    s = embedding(src)
    r = rel_embed(rel)
    o = embedding(dst)
    d = s * r * o
    #score = torch.sum(s * r * o, dim=1)
    #score = torch.norm(s * r * o, p = 1 ,dim=1)
    #e = torch.norm(d, p=1, dim=-1)
    # e2 = -torch.sum(d, dim=0)
    # e3 = -torch.sum(d, dim=-1)
    # e4 = -torch.sum(d, dim=1)
    e1 = -torch.sum(d, dim=-1)

    return e1

class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()
        sigma = 0.2
        self.entity_embedding = nn.Embedding(num_entities, args.node_embed_size)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, args.node_embed_size))
        #self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, args.node_embed_size*args.node_embed_size))

        args.rel_embed = self.relation_embedding
        #args.entity_embedding = self.entity_embedding

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            args.node_embed_size, args.node_embed_size, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            args.node_embed_size, args.node_embed_size, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        x = self.entity_embedding(entity)
        x = F.relu(self.conv1(x, edge_index, edge_type, edge_norm))
        x = F.dropout(x, p = self.dropout_ratio, training = self.training)
        x = self.conv2(x, edge_index, edge_type, edge_norm)
        
        return x

    def pos_embedding(self, embedding, real_data):
        pos_head = embedding[real_data[:, 0]]
        pos_rel = self.relation_embedding[real_data[:, 1]].cuda()
        pos_tail = embedding[real_data[:, 2]]

        return pos_head,pos_rel,pos_tail

    def distmult(self, embedding, triplets, fake_tail1,fake_tail2, fake_head, target, real_triplet):
        s = embedding[triplets[:,0]].cuda()
        r = self.relation_embedding[triplets[:,1]].cuda()
        o = embedding[triplets[:,2]].cuda()

        s_real = embedding[real_triplet[:,0]].cuda()
        r_real = self.relation_embedding[real_triplet[:,1]].cuda()
        o_real = embedding[real_triplet[:,2]].cuda()

        s_real_ratio = torch.empty(0, args.node_embed_size).cuda()
        #r_real_ratio = torch.empty(0, args.node_embed_size*args.node_embed_size).cuda()
        r_real_ratio = torch.empty(0, args.node_embed_size).cuda()
        o_real_ratio = torch.empty(0, args.node_embed_size).cuda()

        for a in range(args.gen_fake_ratio):
            s_real_ratio = torch.cat((s_real_ratio, s_real), dim=0)
            r_real_ratio = torch.cat((r_real_ratio, r_real), dim=0)
            o_real_ratio = torch.cat((o_real_ratio, o_real), dim=0)

        #Change 50% of s_real_ratio to fake head
        #Change 50% of o_real_ratio to fake tail (exclusive of above)

        num_to_generate = len(s_real_ratio)
        choices = np.random.uniform(size=num_to_generate)
        subj = choices > 0.5
        obj = choices <= 0.5
        half_head= s_real_ratio
        half_tail = o_real_ratio

        a = half_head[subj]
        b = half_tail[obj]

        half_head[subj] = fake_head[subj]
        half_tail[obj] = fake_tail1[obj]

        #300 real , 300 randomly replaced , 300 fake
        new_s = torch.cat((s, half_head,s_real_ratio), 0).cuda()
        new_r = torch.cat((r, r_real_ratio, r_real_ratio), 0).cuda()
        new_o = torch.cat((o, half_tail, fake_tail2), 0).cuda()

        a = new_s * new_r * new_o

        # score1 = torch.sum(new_s * new_r * new_o, dim=-1)
        # score2= torch.sum(new_s * new_r * new_o, dim=0)
        # #score3 = torch.sum(new_s * new_r * new_o, dim=2)
        score = torch.sum(new_s * new_r * new_o, dim=-1)
        #score2 = torch.sum(new_s * new_r * new_o, dim=2)

        newtarget = torch.cat((half_tail, fake_tail2), dim=0)
        new_target = torch.zeros(newtarget.size()[0])

        new_target = torch.cat((target, new_target), 0).cuda()

        #Add in fake s & o
        #s has 600 more : 300 fake s and 300 real s
        #o has 600 more : 300 real o and 300 fake o


        #Add in fake triplets

        return score, new_target

    def score_loss(self, embedding, fake_tail1,fake_tail2,fake_head, triplets, target, real_triplet):
        score, new_target = self.distmult(embedding, triplets, fake_tail1, fake_tail2, fake_head, target, real_triplet)



        return F.binary_cross_entropy_with_logits(score, new_target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))

class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)


    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)


    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))

        # If no node features are given, we implement a simple embedding
        # loopkup based on the target node index and its edge type.
        if x_j is None:
            w = w.view(-1, self.out_channels)
            index = edge_type * self.in_channels + edge_index_j
            out = torch.index_select(w, 0, index)
        else:
            w = w.view(self.num_relations, self.in_channels, self.out_channels)
            w = torch.index_select(w, 0, edge_type)
            out = torch.bmm(x_j.unsqueeze(1), w).squeeze(-2)

        return out if edge_norm is None else out * edge_norm.view(-1, 1)

    def update(self, aggr_out, x):
        if self.root is not None:
            if x is None:
                out = aggr_out + self.root
            else:
                out = aggr_out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
