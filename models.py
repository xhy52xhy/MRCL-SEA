#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn
from torch_geometric.utils import softmax
from torch_geometric.nn import global_add_pool
# from torch_scatter import scatter
# from torch_sparse import spmm

from layers import *

from compgcn_conv import CompGCNConv


class Parallel_Co_Attention(nn.Module):
    def __init__(self, hidden_dim, k=300, dropout=0.2, alpha=0.7, device=None):
        super(Parallel_Co_Attention, self).__init__()
        self.alpha = alpha  # 控制全局上下文与原始嵌入之间的平衡系数
        self.W_b = nn.Parameter(torch.zeros(size=(hidden_dim, hidden_dim)))  # 注意力矩阵，用于表示左右嵌入的交互关系
        nn.init.xavier_uniform_(self.W_b.data, gain=1.414)  # Xavier初始化，用于保持矩阵的方差一致性
        self.W_b.requires_grad = True  # W_b需要进行梯度更新

        self.W_l = nn.Parameter(torch.zeros(size=(k, hidden_dim)))  # 左侧嵌入的投影矩阵
        nn.init.xavier_uniform_(self.W_l.data, gain=1.414)  # Xavier初始化
        self.W_l.requires_grad = True  # W_l需要进行梯度更新

        self.W_r = nn.Parameter(torch.zeros(size=(k, hidden_dim)))  # 右侧嵌入的投影矩阵
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)  # Xavier初始化
        self.W_r.requires_grad = True  # W_r需要进行梯度更新

        self.w_hl = nn.Parameter(torch.zeros(size=(1, k)))  # 左侧权重矩阵，用于计算左侧注意力分数
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)  # Xavier初始化
        self.w_hl.requires_grad = True  # w_hl需要进行梯度更新

        self.w_hr = nn.Parameter(torch.zeros(size=(1, k)))  # 右侧权重矩阵，用于计算右侧注意力分数
        nn.init.xavier_uniform_(self.W_r.data, gain=1.414)  # Xavier初始化
        self.w_hr.requires_grad = True  # w_hr需要进行梯度更新

        self.dropout = nn.Dropout(dropout)  # Dropout层，用于防止过拟合
        print("using Parallel")  # 输出提示，表示正在使用并行共注意力机制

    # left_emb:(N * d), right_emb:(T * d)
    def forward(self, left_emb, right_emb):
        N = left_emb.shape[0]  # 左侧嵌入的批量大小
        T = right_emb.shape[0]  # 右侧嵌入的批量大小
        left_emb_T = left_emb.t()  # 左侧嵌入矩阵的转置
        right_emb_T = right_emb.t()  # 右侧嵌入矩阵的转置

        # 计算相似性矩阵，通过右侧嵌入与左侧嵌入交互得到 (T * N)
        affinity_matrix = torch.tanh(torch.matmul(right_emb, torch.matmul(self.W_b, left_emb.t())))

        # 计算左侧的激活表示，将左侧嵌入和右侧嵌入结合，得到 (k * d) 的新表示
        h_left = torch.matmul(self.W_l, left_emb_T) + torch.matmul(torch.matmul(self.W_r, right_emb_T), affinity_matrix)
        h_left = self.dropout(torch.tanh(h_left))  # 应用tanh激活和dropout

        # 计算右侧的激活表示，将右侧嵌入和左侧嵌入结合，得到 (k * d) 的新表示
        h_right = torch.matmul(self.W_r, right_emb_T) + torch.matmul(torch.matmul(self.W_l, left_emb_T),
                                                                     affinity_matrix.t())
        h_right = self.dropout(torch.tanh(h_right))  # 应用tanh激活和dropout

        # 计算左侧和右侧的注意力权重
        l_input = torch.matmul(self.w_hl, h_left)  # 左侧权重 (1 * N)
        r_input = torch.matmul(self.w_hr, h_right)  # 右侧权重 (1 * T)
        a_l = F.softmax(l_input, dim=1)  # 左侧的注意力分布 (1 * N)
        a_r = F.softmax(r_input, dim=1)  # 右侧的注意力分布 (1 * T)

        # 计算全局表示
        global_left = torch.matmul(a_l, left_emb)  # 左侧全局表示 (1 * d)
        global_right = torch.matmul(a_r, right_emb)  # 右侧全局表示 (1 * d)

        # 更新左侧和右侧的嵌入，结合原始嵌入和全局信息，通过alpha控制权重
        left_emb = self.alpha * left_emb + (1 - self.alpha) * torch.matmul(a_l.t().repeat(1, T), right_emb)
        right_emb = self.alpha * right_emb + (1 - self.alpha) * torch.matmul(a_r.t().repeat(1, N), left_emb)

        # 计算全局损失，基于左右全局表示的平方差
        global_loss = torch.sum((global_left - global_right) ** 2, dim=1)

        return global_loss, left_emb, right_emb  # 返回全局损失、更新后的左侧和右侧嵌入

class GraphConvolution1(nn.Module):
    def __init__(self, input_dim, output_dim, featureless=False, act_func=F.relu, residual=True):
        super(GraphConvolution1, self).__init__()
        self.act_func = act_func
        self.residual = residual
        self.featureless = featureless
        if self.residual and input_dim != output_dim:
            self.root = nn.Linear(input_dim, output_dim, False)
            nn.init.xavier_uniform_(self.root.weight)
        if not self.featureless:
            self.linear = nn.Linear(input_dim, output_dim)
            nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, adj, feats):
        device = next(self.parameters()).device  # 获取模型当前的设备
        adj = adj.to(device)  # 确保邻接矩阵在同一设备
        feats = feats.to(device)  # 确保特征矩阵在同一设备

        to_feats = torch.sparse.mm(adj, feats)  # 将邻接矩阵 adj 与特征矩阵 feats 相乘

        degree = torch.sparse.sum(adj, dim=1).to_dense().reshape(-1, 1)
        to_feats = to_feats / degree  # 考虑节点度，进行归一化
        to_feats = to_feats.to(device)

        if not self.featureless:
            to_feats = self.linear(to_feats)  # self.linear 应该已初始化在模型的设备上
            to_feats = to_feats.to(device)

        to_feats = self.act_func(to_feats)  # 激活函数
        to_feats = to_feats.to(device)

        if self.residual:  # 残差连接
            if feats.shape[-1] != to_feats.shape[-1]:
                to_feats = self.root(feats) + to_feats  # 残差连接
                to_feats = to_feats.to(device)
            else:
                to_feats = feats + to_feats  # 残差连接
                to_feats = to_feats.to(device)

        return to_feats


class GAT(nn.Module):
    def __init__(self, n_units, n_heads, dropout, attn_dropout, instance_normalization, diag, use='node'):
        super(GAT, self).__init__()
        self.num_layer = len(n_units) - 1
        self.dropout = dropout
        self.inst_norm = instance_normalization
        if self.inst_norm:
            self.norm = nn.InstanceNorm1d(n_units[0], momentum=0.0, affine=True)
        self.layer_stack = nn.ModuleList()
        self.diag = diag
        self.use = use
        for i in range(self.num_layer):
            f_in = n_units[i] * n_heads[i - 1] if i else n_units[i]
            print("i={} layer: f_in={}, f_out={}".format(i, f_in, n_units[i + 1]))
            self.layer_stack.append(
                MultiHeadGraphAttention(n_heads[i], f_in, n_units[i + 1], attn_dropout, diag, nn.init.ones_, False, use=use))

    def forward(self, x, adj, weight=None):
        if self.inst_norm:
            x = self.norm(x)
        for i, gat_layer in enumerate(self.layer_stack):
            if i + 1 < self.num_layer:
                x = F.dropout(x, self.dropout, training=self.training)
            if self.use == 'node':
                x = gat_layer(x, adj)
            else:
                x = gat_layer(x, adj, weight)
            if self.diag:
                x = x.mean(dim=0)
            if i + 1 < self.num_layer:
                if self.diag:
                    x = F.elu(x)
                else:
                    x = F.elu(x.transpose(0, 1).contiguous().view(adj.size(0), -1))
        if not self.diag:
            x = x.mean(dim=0)

        return x


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout, device=None):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.highway = Highway(nout, nout, device)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # change to leaky relu
        x = F.dropout(x, self.dropout, training=self.training)
        for i in range(0):
            x = self.gc2(x, adj)
        x = self.highway(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_dim, out_dim, device=None):
        super(Highway, self).__init__()
        self.cuda = device
        self.fc1 = self.init_Linear(in_fea=in_dim, out_fea=out_dim, bias=True)
        self.gate_layer = self.init_Linear(in_fea=in_dim, out_fea=out_dim, bias=True)

    def init_Linear(self, in_fea, out_fea, bias):
        linear = nn.Linear(in_features=in_fea, out_features=out_fea, bias=bias)
        if self.cuda is True:
            return linear.cuda()
        else:
            return linear

    def forward(self, x):
        in_fea = x.size(0)
        out_fea = x.size(1)
        normal_fc = torch.tanh(self.fc1(x))
        transformation_layer = torch.sigmoid(self.gate_layer(x))
        carry_layer = 1 - transformation_layer
        allow_transformation = torch.mul(normal_fc, transformation_layer)
        allow_carry = torch.mul(x, carry_layer)
        information_flow = torch.add(allow_transformation, allow_carry)
        return information_flow


class MultiViewEncoder(nn.Module):
    def __init__(self, args, device,
                 ent_num, rel_num, name_size,
                 rel_size, attr_size,
                 use_project_head=False):
        super(MultiViewEncoder, self).__init__()

        self.args = args
        self.device = device
        attr_dim = self.args.attr_dim
        rel_dim = self.args.rel_dim
        dropout = self.args.dropout
        self.ENT_NUM = ent_num
        self.REL_NUM = rel_num
        self.use_project_head = use_project_head

        self.n_units = [int(x) for x in self.args.hidden_units.strip().split(",")]
        self.rel_n_units = [int(x) for x in self.args.rel_hidden_units.strip().split(",")]

        self.n_heads = [int(x) for x in self.args.heads.strip().split(",")]
        self.rel_n_heads = [int(x) for x in self.args.rel_heads.strip().split(",")]

        self.input_dim = int(self.args.hidden_units.strip().split(",")[0])
        self.rel_input_dim = int(self.args.rel_hidden_units.strip().split(",")[0])
        self.ent_dim = 728
        self.rel_output_dim = self.rel_input_dim  # 100

        self.entity_emb = nn.Embedding(self.ENT_NUM, self.input_dim)
        nn.init.normal_(self.entity_emb.weight, std=1.0 / math.sqrt(self.ENT_NUM))
        self.entity_emb.requires_grad = True

        if self.args.structure_encoder == "gcn":
            self.cross_graph_model = GCN(self.n_units[0], self.n_units[1], self.n_units[2],
                                         dropout=self.args.dropout, device=self.args.cuda)
        elif self.args.structure_encoder == "gat":
            self.cross_graph_model = GAT(n_units=self.n_units, n_heads=self.n_heads, dropout=args.dropout,
                                         attn_dropout=args.attn_dropout,
                                         instance_normalization=self.args.instance_normalization, diag=True)
        if self.args.ent_name:
            print("with entity name")
            if self.args.word_embedding == "wc":
                print("name_size:",name_size)
                self.wc_fc = nn.Linear(name_size, self.ent_dim, bias=False)
            if "SRPRS" in self.args.file_dir:
                self.tri_fc = nn.Linear(728, self.ent_dim, bias=False)
            else:
                self.tri_fc = nn.Linear(1156, self.ent_dim, bias=False)
        else:
            if "SRPRS" in self.args.file_dir:
                self.tri_fc = nn.Linear(600, self.ent_dim, bias=False)
            else:
                self.tri_fc = nn.Linear(700, self.ent_dim, bias=False)

        if self.args.w_triple_gat > 0:
            self.rel_emb = nn.Parameter(
                nn.init.sparse_(torch.empty(self.REL_NUM*2, self.rel_input_dim), sparsity=0.15)).to(device)

            self.highway1 = FinalHighway(self.ent_dim).to(device)
            self.highway2 = FinalHighway(self.ent_dim).to(device)
            # self.highway3 = FinalHighway(self.ent_dim).to(device)
            self.ea1 = EALayer(self.REL_NUM, 728, self.rel_output_dim, mode="add", use_ra=False).to(device)
            #EALayer 类是一个关系增强层，它可能使用边的信息（头实体和尾实体）来增强关系的表示。mode 参数控制增强的方式（这里是加法），use_ra 表示是否使用残差连接。
            self.ea2 = EALayer(self.REL_NUM, 728, self.rel_output_dim, mode="add", use_ra=False).to(device)
            # self.ea3 = EALayer(self.REL_NUM, 300, self.rel_output_dim, mode="add", use_ra=self.args.w_ra).to(device)

        # Relation Embedding(for entity)
        self.rel_shared_fc = nn.Linear(self.REL_NUM, rel_dim)#self.REL_NUM 表示关系的数量，rel_dim 表示关系嵌入的维度。通过一个线性层将关系映射到一个低维空间，这是关系表示的一种简单形式。
        # self.emb_rel_fc = nn.Linear(200 * 2, 200)
        # Attribution Embedding(for entity)
        if self.args.w_attr:
            self.att_fc = nn.Linear(attr_size, attr_dim)

        if self.use_project_head:
            # self.img_pro = ProjectionHead(img_dim, img_dim, img_dim, dropout)
            self.att_pro = ProjectionHead(attr_dim, attr_dim, attr_dim, dropout)
            self.rel_pro = ProjectionHead(400, 400, 400, dropout)
            self.gph_pro = ProjectionHead(self.n_units[2], self.n_units[2], self.n_units[2], dropout)

        self.fusion = MultiViewFusion(modal_num=self.args.inner_view_num,
                                      with_weight=self.args.with_weight)

    def forward(self,
                input_idx, r_input_idx,
                e_in, e_out, e_adj, epoch,
                r_in_adj, r_out_adj,r_path_adj,
                edge_index_all, rel_all, name_emb,
                rel_features_in=None,
                rel_features_out=None,
                att_features=None):

        if self.args.w_gph:
            gph_emb = self.cross_graph_model(self.entity_emb(input_idx), e_adj)
            # gph_emb = self.cross_graph_model(name_emb, e_adj)
        else:
            gph_emb = None

        if self.args.w_triple_gat:
            rel_emb = self.rel_emb

        if self.args.w_rel:
            rel_in_f = self.rel_shared_fc(rel_features_in)
            rel_out_f = self.rel_shared_fc(rel_features_out)
            ent_rel_features = torch.cat([rel_in_f, rel_out_f], dim=1)
            ent_rel_emb = ent_rel_features
            # ent_rel_emb = self.emb_rel_fc(ent_rel_features)
        else:
            ent_rel_emb = None

        if self.args.w_attr:
            att_emb = self.att_fc(att_features)
        else:
            att_emb = None
        if self.args.ent_name > 0 and self.args.word_embedding == "wc":
            name_emb = self.wc_fc(name_emb)
            # name_emb=name_emb

        joint_emb = self.fusion([name_emb, gph_emb, ent_rel_emb, att_emb])
        # print("name_emb:", name_emb)
        # print("gph_emb:", gph_emb.size())
        # print("ent_rel_emb:", ent_rel_emb.size())
        # print("att_emb:", att_emb.size())
        if self.args.w_triple_gat:

            # print("joint_emb:",joint_emb.size()).
            joint_emb = self.tri_fc(joint_emb)
            res_att = None
            # print("joint_emb, edge_index_all, rel_all, rel_emb, res_att:",joint_emb.size(),edge_index_all.size(),rel_all.size())
            x_e, rel_emb, res_att = self.ea1(joint_emb, edge_index_all, rel_all, rel_emb, res_att)
            # print("joint_emb.size(),x_e.size()",joint_emb.size(),x_e.size())
            # exit()
            x_e_1 = self.highway1(joint_emb, x_e)
            # x_e_1 = x_e

            x_e, rel_emb, res_att = self.ea2(x_e_1, edge_index_all, rel_all, rel_emb, res_att)
            x_e_2 = self.highway2(x_e_1, x_e)
            # x_e_2 = x_e
            # x_e, rel_emb, res_att = self.ea3(x_e_2, edge_index_all, rel_all, rel_emb, res_att)
            # x_e_3 = self.highway3(x_e_2, x_e)
            tri_emb = torch.cat([x_e_1, x_e_2], dim=1) 
            # tri_emb = x_e_1
        else:
            rel_emb = None
            tri_emb = None

        return gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb


def pooling(ent_rel_list, method="avg"):
    # len = ent_rel_list.shape[0]
    if method == "avg":
        return torch.mean(ent_rel_list, dim=0).unsqueeze(0)
    elif method == 'max':
        return torch.max(ent_rel_list, 0)[0].unsqueeze(0)
    elif method == 'min':
        return torch.min(ent_rel_list, 0)[0].unsqueeze(0)


class MultiViewFusion(nn.Module):
    def __init__(self, modal_num, with_weight=1):
        super().__init__()
        self.modal_num = modal_num
        self.requires_grad = True if with_weight > 0 else False
        self.weight = nn.Parameter(torch.ones((self.modal_num, 1)),
                                   requires_grad=self.requires_grad)

    def forward(self, embs):
        assert len(embs) == self.modal_num
        weight_norm = F.softmax(self.weight, dim=0)
        # for idx, emb in enumerate(embs):
            # if emb is not None:
                # print(f"embs[{idx}] 的形状: {emb.shape}")
        embs = [weight_norm[idx] * F.normalize(embs[idx]) for idx in range(self.modal_num) if embs[idx] is not None]
        joint_emb = torch.cat(embs, dim=1)
        return joint_emb


class FinalHighway(nn.Module):
    def __init__(self, x_hidden):
        super(FinalHighway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2) + torch.mul(1 - gate, x1)
        return x


class EALayer(nn.Module):
    def __init__(self, rel_num, e_hidden, r_hidden, mode="add", use_ra=False):
        super(EALayer, self).__init__()
        print("using EALayer, mode={}".format(mode))  # 输出正在使用的模式
        self.use_ra = use_ra  # 是否使用 RALayer（关系注意力层）

        self.mode = mode  # 模式类型（add、sub、cross、concat 等）
        self.ww = nn.Linear(r_hidden, e_hidden, bias=False)  # 将关系嵌入从 r_hidden 映射到 e_hidden
        self.rel = nn.Linear(r_hidden, r_hidden, bias=False)  # 关系嵌入的线性变换层

        if self.use_ra:
            self.ra_layer = RALayer(e_hidden=e_hidden, r_hidden=r_hidden)  # 如果使用关系注意力层，初始化 RALayer

        if self.mode == "cross":
            self.rel_weight = nn.Parameter(nn.init.xavier_normal_(torch.empty(2*rel_num, e_hidden)))  # 如果模式是 cross，初始化关系权重矩阵
        elif self.mode == "concat":
            self.cat_fc = nn.Linear(e_hidden, e_hidden*2, bias=False)  # 如果模式是 concat，先扩展维度
            self.out_fc = nn.Linear(e_hidden*2, e_hidden, bias=False)  # 再通过线性层输出
        else:
            pass

    def forward(self, x, edge_index, edge_type, rel_emb, res_att):
        if self.use_ra:
            # 如果使用了 RALayer，调用 RALayer 来更新关系嵌入和注意力
            rel_emb, res_att = self.ra_layer(x, edge_index, edge_type, rel_emb, res_att)

        r_emb = self.ww(rel_emb)  # 通过线性层将关系嵌入 rel_emb 转换为与实体嵌入一致的维度300

        edge_index_i = edge_index[0]  # 边的起点索引（头实体索引）(2,84556)

        edge_index_j = edge_index[1]  # 边的终点索引（尾实体索引）

        e_head = x[edge_index_i]  # 从实体嵌入中获取头实体嵌入(84556,300)

        e_rel = r_emb[edge_type]  # 获取对应的关系嵌入


        # 根据不同模式进行头实体和关系嵌入的组合
        if self.mode == "add":
            h_r = e_head + e_rel  # 模式 "add"：头实体与关系嵌入相加
        elif self.mode == "sub":
            h_r = e_head - e_rel  # 模式 "sub"：头实体与关系嵌入相减
        elif self.mode == "cross":
            rel_weight = torch.index_select(self.rel_weight, 0, edge_type)  # 模式 "cross"：选择相应的关系权重
            h_r = e_head * e_rel * rel_weight + e_head * rel_weight  # 对头实体和关系嵌入做逐元素乘积，并加上头实体和关系权重的乘积
        elif self.mode == "concat":
            x = self.cat_fc(x)  # 模式 "concat"：扩展实体嵌入维度
            h_r = torch.cat([e_head, e_rel], dim=1)  # 将头实体和关系嵌入拼接在一起
        else:
            pass
#
# embs[1] 的形状: torch.Size([31437, 128])
# embs[2] 的形状: torch.Size([31437, 200])
# e_head.size() torch.Size([84556, 300])
# e_rel.size() torch.Size([84556, 300])
# e_tail.size() torch.Size([84556, 300])
# dp_att.size() torch.Size([84556])
# attention_weights.size() torch.Size([84556])
# weighted_h_r.size(),edge_index_j.size() torch.Size([84556, 300]) torch.Size([84556])
# x_e.size() torch.Size([24944, 300])
# joint_emb.size(),x_e.size() torch.Size([31437, 300]) torch.Size([24944, 300])

        e_tail = x[edge_index_j]  # 获取尾实体嵌入
        dp_att = torch.sum(h_r * e_tail, dim=-1)  # 计算头实体、关系嵌入和尾实体之间的点积注意力得分
        attention_weights = torch.softmax(dp_att, dim=-1)  # 计算归一化的注意力权重
        # 计算经过注意力加权后的更新嵌入，使用 scatter 将更新的嵌入聚合到尾实体
        weighted_h_r = h_r * attention_weights.unsqueeze(dim=-1)  # 扩展 attention_weights 的维度
        x_e = global_add_pool(weighted_h_r, edge_index_j, size=None)  # 使用 edge_index_j 进行聚合
        # x_e = scatter(h_r * torch.unsqueeze(attention_weights, dim=-1), edge_index_j, dim=0, reduce='sum')
        if self.mode == "concat":
            x_e = self.out_fc(x_e)  # 如果模式为 concat，使用线性层将维度缩回

        x_e = F.relu(x_e)  # 通过 ReLU 激活函数
        return x_e, self.rel(rel_emb), res_att  # 返回更新后的实体嵌入、关系嵌入和残差注意力


class MultiLayerGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layer, dropout_rate=0.5, featureless=True, residual=False):
        super(MultiLayerGCN, self).__init__()
        self.residual = residual
        self.dropout = nn.Dropout(dropout_rate)
        self.gcn_list = nn.ModuleList()
        assert num_layer >= 1
        dim = in_dim
        for i in range(num_layer - 1):
            if i == 0:
                self.gcn_list.append(GraphConvolution1(dim, out_dim, featureless))
                dim = out_dim
            else:
                self.gcn_list.append(GraphConvolution1(out_dim, out_dim, False))
        self.gcn_list.append(
            GraphConvolution1(dim, out_dim, False))
    def preprocess_adj(self, edges):
        device = next(self.parameters()).device
        edges = torch.cat((edges, edges.flip(dims=[1, ])), dim=0)  # shape=[E * 2, 2]
        # print('edges.shape',edges.shape[0],'edges.transpose(0, 1).shape:',edges.transpose(0, 1).shape)
        adj = torch.sparse.FloatTensor(edges.transpose(0, 1), torch.ones(edges.shape[0], device=device))
        # print('adj1.shape', adj.shape)
        M, N = adj.shape
        assert M == N
        # if not self.residual:
        self_loop = torch.arange(N, device=device).reshape(-1, 1).repeat(1, 2)  # shape = [N, 2]

        self_loop = torch.sparse.FloatTensor(self_loop.transpose(0, 1),
                                             torch.ones(self_loop.shape[0], device=device))
        adj = adj + self_loop
        # print('adj2.shape', adj.shape)
        adj = adj.coalesce()
        # print('adj3.shape', adj.shape)
        torch.clamp_max_(adj._values(), 1)
        # print('adj4.shape', adj.shape)
        return adj
    # def preprocess_adj(self, edges):
    #     device = next(self.parameters()).device
    #
    #     # 获取边的节点数
    #     unique_nodes = torch.unique(edges)  # 获取所有独立节点编号
    #     node_count = unique_nodes.shape[0]  # 实际节点数量
    #
    #     # 建立节点编号映射，将实际的节点编号映射到从 0 开始的连续编号
    #     node_map = {node.item(): idx for idx, node in enumerate(unique_nodes)}
    #
    #     edges_mapped = torch.tensor([[node_map[e[0].item()], node_map[e[1].item()]] for e in edges], device=device)
    #     edges_mapped = torch.cat((edges_mapped, edges_mapped.flip(dims=[1, ])), dim=0)  # shape=[E * 2, 2]
    #     adj = torch.sparse.FloatTensor(edges_mapped.transpose(0, 1), torch.ones(edges_mapped.shape[0], device=device),
    #                                    torch.Size([node_count, node_count]))  # 设置稀疏矩阵的维度为实际节点数量
    #     self_loop = torch.arange(node_count, device=device).reshape(-1, 1).repeat(1, 2)  # shape = [node_count, 2]
    #     self_loop = torch.sparse.FloatTensor(self_loop.transpose(0, 1),
    #                                          torch.ones(self_loop.shape[0], device=device),
    #                                          torch.Size([node_count, node_count]))
    #     adj = adj + self_loop
    #     adj = adj.coalesce()
    #     torch.clamp_max_(adj._values(), 1)

    #     return adj

    def forward(self, edges, graph_embedding):#(edges)只保留头为实体
        # print('edges.shape:',edges.shape)#edges.shape: torch.Size([45986, 2])
        adj = self.preprocess_adj(edges)
        # print('adj.shape',adj.shape)
        # print(adj.shape,graph_embedding.shape)#torch.Size([4012, 4012]) torch.Size([5576, 768])  torch.Size([25500, 25500]) torch.Size([19388, 768])
        for gcn in self.gcn_list:
            graph_embedding = self.dropout(graph_embedding)
            graph_embedding = gcn(adj, graph_embedding)
        return graph_embedding

class NameGCN(nn.Module):
    def __init__(self, dim, layer_num, drop_out, sr_ent_embed, tg_ent_embed, edges_sr, edges_tg):
        super(NameGCN, self).__init__()
        self.embedding_sr = nn.Parameter(sr_ent_embed, requires_grad=False) #sr_embed.shape torch.Size([19388, 768])
        self.embedding_tg = nn.Parameter(tg_ent_embed, requires_grad=False)
        self.edges_sr = nn.Parameter(edges_sr, requires_grad=False)
        self.edges_tg = nn.Parameter(edges_tg, requires_grad=False)
        in_dim = sr_ent_embed.shape[1]
        # print(sr_ent_embed.size(),in_dim,dim)#torch.Size([19388, 768]) 768 372
        self.gcn = MultiLayerGCN(in_dim, dim, layer_num, drop_out, featureless=False, residual=False)

    def forward(self,):
        # print('self.edges_sr, self.embedding_sr:',self.edges_sr.shape, self.embedding_sr.shape)
        #self.edges_sr, self.embedding_sr: torch.Size([2483, 2]) torch.Size([3986, 768])
        # print('eself.edges_sr:',self.edges_sr)
        sr_ent_hid = self.gcn(self.edges_sr, self.embedding_sr)#edges_sr(只保留头尾实体),embedding_sr.shape torch.Size([19388, 768])
        tg_ent_hid = self.gcn(self.edges_tg, self.embedding_tg)
        return sr_ent_hid, tg_ent_hid