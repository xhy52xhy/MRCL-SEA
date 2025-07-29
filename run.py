#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os

import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import gc
from pprint import pprint
import torch.optim as optim
from torch_geometric.utils import sort_edge_index
from lion_optim import *
from loss import *
from load import *
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from load_data import *
# from encode import *
def train():
    print("[start training...] ")


class SCMEA:
    def __init__(self):

        self.nhop_path = None
        self.rel_ht_dict = None
        self.parallel_co_attention = None
        self.nca_loss = None
        self.ent_name_dim = None
        self.ent_features = None
        self.triples_2 = None
        self.triples_1 = None
        self.line_graph_index_in = None
        self.line_graph_index_out = None
        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.r_hs = None
        self.r_ts = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None
        self.long_tail_en = []
        self.edge_index_all = None
        self.rel_all = None
        self.rel_features_in = None
        self.rel_features_out = None
        self.att_features = None

        self.left_non_train = None
        self.right_non_train = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.e_adj = None
        self.r_adj = None
        self.r_in_adj = None
        self.r_out_adj = None
        self.r_path_adj = None
        self.train_ill = None
        self.test_ill_ = None
        self.test_ill = None
        self.test_left = None
        self.test_right = None
        self.e_in = None
        self.e_out = None

        self.multiview_encoder = None
        self.sim_module = None

        self.gcn_pro = None
        self.rel_pro = None
        self.attr_pro = None
        self.img_pro = None

        self.input_dim = None
        self.entity_emb = None
        self.e_input_idx = None
        self.r_input_idx = None
        self.n_units = None
        self.n_heads = None
        self.cross_graph_model = None
        self.params = None
        self.optimizer = None
        self.all_duplicates=None
        self.loss = None
        self.cur_max_epoch = 0

        self.multi_loss_layer = None
        self.align_multi_loss_layer = None
        self.fusion = None  # fusion module

        self.parser = argparse.ArgumentParser()
        self.args = self.parse_options(self.parser)

        self.set_seed(self.args.seed, self.args.cuda)

        self.device = torch.device("cuda:0" if self.args.cuda and torch.cuda.is_available() else "cpu")


        self.init_data()
        self.init_emb()
        self.init_model()

    @staticmethod
    def parse_options(parser):
        parser.add_argument("--file_dir", type=str, default="data/DBP15K/zh_en", required=False,
                            help="input dataset file directory, ('data/DBP15K/zh_en', 'data/DWY100K/dbp_wd')")
        parser.add_argument("--rate", type=float, default=0.3
                            , help="training set rate")
        parser.add_argument("--cuda", action="store_true", default=True, help="whether to use cuda or not")
        parser.add_argument("--seed", type=int, default=2024, help="random seed")
        parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
        parser.add_argument("--check_point", type=int, default=100, help="check point")
        parser.add_argument("--check_point_il", type=int, default=50, help="check point after il")
        parser.add_argument("--hidden_units", type=str, default="128,128,128",
                            help="hidden units in each hidden layer(including in_dim and out_dim), splitted with comma")
        parser.add_argument("--rel_hidden_units", type=str, default="100,100,100",
                            help="hidden units in each rel hidden layer(including in_dim and out_dim)")
        parser.add_argument("--heads", type=str, default="2,2", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--rel_heads", type=str, default="1", help="heads in each gat layer, splitted with comma")
        parser.add_argument("--instance_normalization", action="store_true", default=False,
                            help="enable instance normalization")
        parser.add_argument("--lr", type=float, default=0.00005, help="initial learning rate")
        parser.add_argument("--weight_decay", type=float, default=0.01, help="weight decay (L2 loss on parameters)")
        parser.add_argument("--dropout", type=float, default=0.0, help="dropout rate for layers")
        parser.add_argument("--attn_dropout", type=float, default=0.0, help="dropout rate for gat layers")
        parser.add_argument("--dist", type=int, default=2, help="L1 distance or L2 distance. ('1', '2')")
        parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference")
        parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
        parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
        parser.add_argument("--semi_learn_step", type=int, default=10, help="If IL, what's the update step?")
        parser.add_argument("--il_start", type=int, default=500, help="If Il, when to start?")
        parser.add_argument("--bsize", type=int, default=7500, help="batch size")
        parser.add_argument("--unsup", action="store_true", default=False)
        parser.add_argument("--lta_split", type=int, default=0, help="split in {0,1,2,3,|splits|-1}")
        parser.add_argument("--with_weight", type=int, default=1, help="Whether to weight the fusion of different "
                                                                       "modal features")
        parser.add_argument("--structure_encoder", type=str, default="gcn", help="the encoder of structure view, "
                                                                                 "[gcn|gat]")
        parser.add_argument("--rel_structure_encoder", type=str, default="gcn", help="the encoder of relation view, "
                                                                                     "[gcn|gat]")
        parser.add_argument("--w_triple_gat", type=int, default=1, help="Whether to use the W-Triple GAT")
        parser.add_argument("--optimizer", type=str, default="AdamW", help="AdamW | Lion")
        parser.add_argument("--cl", action="store_true", default=True, help="CL")
        parser.add_argument("--tau", type=float, default=0.1, help="the temperature factor of contrastive loss")
        parser.add_argument("--ab_weight", type=float, default=0.5, help="the weight of NTXent Loss")
        parser.add_argument("--attr_dim", type=int, default=100, help="the hidden size of attr and rel features")
        parser.add_argument("--rel_dim", type=int, default=100, help="the hidden size of relation feature")
        parser.add_argument("--w_gph", action="store_false", default=True, help="with gph features")
        parser.add_argument("--w_rel", action="store_false", default=True, help="with rel features")
        parser.add_argument("--w_attr", action="store_false", default=True, help="with attr features")
        parser.add_argument("--w_in_gph", type=int, default=1, help="interactive_learning with gph features")
        parser.add_argument("--w_in_rel", type=int, default=1, help="interactive_learning with rel features")
        parser.add_argument("--w_in_att", type=int, default=1, help="interactive_learning with att features")
        parser.add_argument("--expend_t", type=float, default=0.3, help="expend entities before training")
        parser.add_argument("--pro_lte", type=float, default=0.3, help="The proportion of long tail entities")
        parser.add_argument("--w_lg", type=int, default=0, help="with lg features")
        parser.add_argument("--w_ra", type=int, default=0, help="with ra?")
        parser.add_argument("--inner_view_num", type=int, default=4, help="the number of inner view")
        parser.add_argument("--loss", type=str, default="nca", help="[nca|hinge]")
        parser.add_argument("--gamma", type=float, default=3, help="expend entities before training")
        parser.add_argument("--word_embedding", type=str, default="wc", help="the type of name embedding, "
                                                                                "[glove|wc]")
        parser.add_argument("--ent_name", type=int, default=1, help="init with entity name")
        parser.add_argument("--use_project_head", action="store_true", default=True, help="use projection head")

        parser.add_argument("--zoom", type=float, default=0.1, help="narrow the range of losses")
        parser.add_argument("--reduction", type=str, default="mean", help="[sum|mean]")
        parser.add_argument("--save_path", type=str, default="../save", help="save path")
        return parser.parse_args()

    @staticmethod
    def set_seed(seed, cuda=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def init_data(self):
        # Load data
        kg_list = [1, 2]
        file_dir = self.args.file_dir
        device = self.device
        self.ent2id_dict, self.all_duplicates,self.ills, self.triples_1, self.triples_2, self.triples, self.ids, self.rel_ht_dict = read_raw_data(
            file_dir, kg_list)
        e1 = os.path.join(file_dir, 'ent_ids_1')
        e2 = os.path.join(file_dir, 'ent_ids_2')
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)
        self.ENT_NUM = len(self.ent2id_dict)+len(self.all_duplicates)
        self.REL_NUM = len(self.rel_ht_dict)
        np.random.shuffle(self.ills)
        self.train_ill = np.array(self.ills[:int(len(self.ills) // 1 * self.args.rate)], dtype=np.int32)
        self.test_ill_ = self.ills[int(len(self.ills) // 1 * self.args.rate):]
        self.test_ill = np.array(self.test_ill_, dtype=np.int32)
        head_l = []
        rel_l = []
        tail_l = []
        for (head, rel, tail) in self.triples:
            head_l.append(head)
            rel_l.append(rel)
            tail_l.append(tail)
        head_l = torch.tensor(head_l, dtype=torch.long)
        rel_l = torch.tensor(rel_l, dtype=torch.long)
        # print(rel_l.max())
        tail_l = torch.tensor(tail_l, dtype=torch.long)

        edge_index = torch.stack([head_l, tail_l], dim=0)

        edge_index, rel = sort_edge_index(edge_index, rel_l)
        edge_index_all = torch.cat([edge_index, edge_index[[1, 0]]], dim=1)
        rel_all = torch.cat([rel, rel + rel.max() + 1])

        self.edge_index_all = edge_index_all.to(device)
        self.rel_all = rel_all.to(device)

        self.test_left = torch.LongTensor(self.test_ill[:, 0].squeeze()).to(device)
        self.test_right = torch.LongTensor(self.test_ill[:, 1].squeeze()).to(device)

        self.left_non_train = list(set(self.left_ents) - set(self.train_ill[:, 0].tolist()))
        self.right_non_train = list(set(self.right_ents) - set(self.train_ill[:, 1].tolist()))

        print("-----dataset summary-----")
        print("dataset:\t", file_dir)
        print("triple num:\t", len(self.triples))
        print("entity num:\t", self.ENT_NUM)
        print("relation num:\t", self.REL_NUM)
        print("#left entity : %d, #right entity: %d" % (len(self.left_ents), len(self.right_ents)))
        print("train ill num:\t", self.train_ill.shape[0], "\ttest ill num:\t", self.test_ill.shape[0])
        print("-------------------------")

    # def init_emb(self):
    #     file_dir = self.args.file_dir
    #     device = self.device
    #     if self.args.word_embedding == "glove" and self.args.ent_name > 0:
    #         if "SRPRS" in file_dir:
    #             vec_path = file_dir + '/' + file_dir.split('/')[-1] + '_word.npy'
    #             with open(vec_path, 'rb') as f:
    #                 ent_features = np.load(f)
    #                 ent_features = torch.Tensor(ent_features)
    #         else:
    #             word2vec_path = file_dir + '/' + file_dir.split('/')[-1].split('_')[0] + '_vectorList.json'
    #             with open(word2vec_path, 'r', encoding='utf-8') as f:
    #                 ent_features = torch.tensor(json.load(f))
    #         ent_features.requires_grad = True
    #         self.ent_features = torch.Tensor(ent_features).to(device)
    #         self.ent_name_dim = self.ent_features.shape[1]
    #     if self.args.word_embedding == "wc" and self.args.ent_name > 0:
    #         vec_path = file_dir + '/' + file_dir.split('/')[-1] + '_wc.npy'
    #         with open(vec_path, 'rb') as f:
    #             ent_features = np.load(f)
    #         self.ent_features = torch.Tensor(ent_features).to(device)
    #         self.ent_features.requires_grad = True
    #         self.ent_name_dim = self.ent_features.shape[1]
    #     a1 = os.path.join(file_dir, 'training_attrs_1')
    #     a2 = os.path.join(file_dir, 'training_attrs_2')
    #     self.att_features = load_attr([a1, a2], self.ENT_NUM, self.ent2id_dict, 1000)
    #     self.att_features = torch.Tensor(self.att_features).to(device)
    #     self.rel_features_in, self.rel_features_out = load_relation(self.ENT_NUM, self.REL_NUM, self.triples)
    #     self.rel_features_in = torch.Tensor(self.rel_features_in).to(device)
    #     self.rel_features_out = torch.Tensor(self.rel_features_out).to(device)
    #     self.e_adj = get_adjr(self.ENT_NUM, self.triples, norm=True)  # getting a sparse tensor r_adj
    #     self.e_adj = self.e_adj.to(self.device)
    #
    def init_word_emb(self,load):

        id2entity_sr = sorted(load.id2entity_sr.items(), key=lambda x: x[0])
        sr_text = [x[1] for x in id2entity_sr]
        id2entity_tg = sorted(load.id2entity_tg.items(), key=lambda x: x[0])
        tg_text = [x[1] for x in id2entity_tg]

        return sr_text,tg_text
    def init_emb(self):
        train_seeds_ratio=0.3
        self.directory='data/DBP15K/zh_en'
        nega_sample_num=25
        name_channel = True
        attribute_value_channel = False
        literal_attribute_channel = False
        digit_attribute_channel = False
        load_new_seed_split = False

        # loaded_data=LoadData(train_seeds_ratio,self.directory,nega_sample_num,name_channel,attribute_value_channel,digit_attribute_channel,load_new_seed_split,self.device)
        # sr_text,tg_text=self.init_word_emb(loaded_data)
        # # # print('device2:',device)
        # bert_model = BGE()
        # sr_text_emb = bert_model.pooled_encode_batched(sr_text)
        # tg_text_emb=bert_model.pooled_encode_batched(tg_text)
        # #
        # print('sr_text_emb.size(),tg_text_emb.size():',sr_text_emb.size(),tg_text_emb.size())
        # sr_ent_emb, tg_ent_emb = sr_text_emb.detach().cpu().numpy(), tg_text_emb.detach().cpu().numpy()
        # # # # 保存为 NumPy 的 .npy 文件
        # np.save('data/DBP15K/zh_en/ent_future/sr_features_bge.npy', sr_text_emb)
        # np.save('data/DBP15K/zh_en/ent_future/tg_features_bge.npy', tg_text_emb)
        # exit()
        # sr_ent_embed=np.load('data/DBP15K/zh_en/sr_ent_features.npy')
        # tg_ent_embed = np.load('data/DBP15K/zh_en/tg_ent_features.npy')
        # sr_ent_embed, tg_ent_embed = torch.tensor(sr_ent_embed).to(device), torch.tensor(tg_ent_embed).to(device)
        # print_time_info('Begin preprocessing adjacent matrix')
        # edges_sr = torch.tensor(loaded_data.triples_sr)[:, [0, 2]]  # (只保留头尾实体)
        # edges_tg = torch.tensor(loaded_data.triples_tg)[:, [0, 2]]
        # edges_sr = torch.unique(edges_sr, dim=0)  # 二元组去重
        # edges_tg = torch.unique(edges_tg, dim=0)
        # sr_ent_embed=loaded_data.sr_embed
        # tg_ent_embed=loaded_data.tg_embed
        sr_ent_embed,tg_ent_embed= np.load('data/DBP15K/zh_en/sr_features_bge.npy'),np.load('data/DBP15K/zh_en/tg_features_bge.npy')
        sr_ent_embed, tg_ent_embed = torch.tensor(sr_ent_embed).to(self.device), torch.tensor(tg_ent_embed).to(self.device)
        # sr_ent_embed, tg_ent_embed = torch.tensor(sr_ent_embed).to(self.device), torch.tensor(tg_ent_embed).to(self.device)
        print('sr_ent_embed.size(),tg_ent_embed.size()',sr_ent_embed.size(),tg_ent_embed.size())        #sr_ent_embed.size(),tg_ent_embed.size() torch.Size([24944, 768]) torch.Size([6493, 768])

        # name_gcn = NameGCN(dim=372, layer_num=2, drop_out=0.0, sr_ent_embed=sr_ent_embed, tg_ent_embed=tg_ent_embed,
        #                    edges_sr=edges_sr, edges_tg=edges_tg)

        # sr_ent_emb, tg_ent_emb = sr_ent_embed.detach().cpu().numpy(), tg_ent_embed.detach().cpu().numpy()
        # sr_ent_hid, tg_ent_hid=name_gcn()
        name_emb=torch.cat((sr_ent_embed,tg_ent_embed),dim=0)
        # name_emb=np.load('data/DBP15K/zh_en/ent_emb.npy')

        # name_emb = np.load('../../glove_encoding/zh_en_wc.npy')

        name_emb=torch.tensor(name_emb).to(self.device).float()

        #已完成工作：对名字编码。
        # #但是读取文件的包含超链接的函数得更改一下。
        # #下一步就是合并join编码返回。
        file_dir = self.args.file_dir


        # ent_features_np = self.ent_features.cpu().numpy()
        # # 保存为 NumPy 的 .npy 文件
        # np.save('data/DBP15K/zh_en/ent_features.npy', ent_features_np)
        # self.ent_features = np.load('data/DBP15K/zh_en/ent_features.npy')
        # 在创建张量时设置 requires_grad=True，并直接将其移动到设备
        self.ent_features = torch.tensor(name_emb, device=self.device, requires_grad=True)
        self.ent_name_dim = self.ent_features.shape[1]
        a1 = os.path.join(file_dir, 'training_attrs_1')
        a2 = os.path.join(file_dir, 'training_attrs_2')
        # print('(len( self.ENT_NUM),'+',len(self.ent2id_dict):',( self.ENT_NUM),'+',len(self.ent2id_dict))
        self.att_features = load_attr([a1, a2], self.ENT_NUM, self.ent2id_dict, 1000)
        self.att_features = torch.Tensor(self.att_features).to(self.device)
        self.rel_features_in, self.rel_features_out = load_relation(self.ENT_NUM, self.REL_NUM, self.triples)
        self.rel_features_in = torch.Tensor(self.rel_features_in).to(self.device)
        self.rel_features_out = torch.Tensor(self.rel_features_out).to(self.device)
        self.e_adj = get_adjr(self.ENT_NUM, self.triples, norm=True)  # getting a sparse tensor r_adj
        self.e_adj = self.e_adj.to(self.device)

    def init_model(self):
        rel_size = self.rel_features_in.shape[1]
        attr_size = self.att_features.shape[1]

        self.multiview_encoder = MultiViewEncoder(args=self.args, device=self.device,
                                                  ent_num=self.ENT_NUM,
                                                  rel_num=self.REL_NUM,
                                                  name_size=self.ent_name_dim,
                                                  rel_size=rel_size, attr_size=attr_size,
                                                  use_project_head=False).to(self.device)
        
        # self.parallel_co_attention = Parallel_Co_Attention(hidden_dim=300)
        self.params = [
            {"params":
                 list(self.multiview_encoder.parameters())
                 # list(self.parallel_co_attention.parameters())
             }]

        if self.args.loss == "nca":
            # multi-view loss
            print("using NCA loss")
            self.gcn_nca_loss = NCA_loss(alpha=3, beta=6, ep=0.0, device=self.device)
            self.nca_loss = NCA_loss(alpha=15, beta=10, ep=0.0, device=self.device)
            if self.args.cl:
                print("CL!")
                self.loss = BiCl(device=self.device, tau=self.args.tau, ab_weight=self.args.ab_weight, n_view=2)
        else:
            print("using Hinge loss")
            # self.loss = L1_Loss(self.args.gamma)

        # select optimizer
        if self.args.optimizer == "AdamW":
            self.optimizer = optim.AdamW(
                self.params,
                lr=self.args.lr
            )
        else:
            self.optimizer = Lion(
                self.params,
                lr=self.args.lr
            )

        print("--------------------model details--------------------")
        print("SCMEA model details:")
        print(self.multiview_encoder)
        print("optimiser details:")
        print(self.optimizer)

    def semi_supervised_learning(self, epoch):
        with torch.no_grad():
            gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb = self.multiview_encoder(self.e_input_idx,
                                                                                                self.r_input_idx,
                                                                                                self.e_in, self.e_out,
                                                                                                self.e_adj, epoch,
                                                                                                self.r_in_adj,
                                                                                                self.r_out_adj,
                                                                                                self.r_path_adj,
                                                                                                self.edge_index_all,
                                                                                                self.rel_all,
                                                                                                self.ent_features,
                                                                                                self.rel_features_in,
                                                                                                self.rel_features_out,
                                                                                                self.att_features)

            if tri_emb is not None:
                tri_emb = F.normalize(tri_emb)
            
            joint_emb = F.normalize(joint_emb)

        distance_list = []
        d_f = None
        for i in np.arange(0, len(self.left_non_train), 1000):
            d_joi = pairwise_distances(joint_emb[self.left_non_train[i:i + 1000]], joint_emb[self.right_non_train])
            if tri_emb is not None:
                d_tri = pairwise_distances(tri_emb[self.left_non_train[i:i + 1000]], tri_emb[self.right_non_train])
                d_f = d_joi + d_tri
            else:
                d_f = d_joi
            distance_list.append(d_f)
        distance = torch.cat(distance_list, dim=0)
        preds_l = torch.argmin(distance, dim=1).cpu().numpy().tolist()
        preds_r = torch.argmin(distance.t(), dim=1).cpu().numpy().tolist()
        del distance_list, distance
        del gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb
        return preds_l, preds_r

    # Train
    def train(self):
        args = self.args
        pprint(args)
        print("[start training...] ")
        new_links = []
        max_hit1 = .0
        epoch_CG = 0
        bsize = self.args.bsize
        device = self.device

        self.e_input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(device)#self.ENT_NUM 39654
        #self.e_input_idx   tensor([    0,     1,     2,  ..., 39651, 39652, 39653], device='cuda:0')
        self.r_input_idx = torch.LongTensor(np.arange(self.REL_NUM)).to(device)
        for epoch in range(self.args.epochs):
            if self.args.optimizer == "AdamW" and epoch == epoch >= self.args.il_start:
                self.optimizer = optim.AdamW(self.params, lr=self.args.lr / 5)

            t_epoch = time.time()
            self.multiview_encoder.train()

            self.optimizer.zero_grad()

            gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb = self.multiview_encoder(self.e_input_idx,
                                                                                                self.r_input_idx,
                                                                                                self.e_in, self.e_out,
                                                                                                self.e_adj, epoch,
                                                                                                self.r_in_adj,
                                                                                                self.r_out_adj,
                                                                                                self.r_path_adj,
                                                                                                self.edge_index_all,
                                                                                                self.rel_all,
                                                                                                self.ent_features,
                                                                                                self.rel_features_in,
                                                                                                self.rel_features_out,
                                                                                                self.att_features)

            loss_sum_all, loss_tri_all, loss_attr_all, loss_joi_all = 0, 0, 0, 0
            epoch_CG += 1

            np.random.shuffle(self.train_ill)
            for si in np.arange(0, self.train_ill.shape[0], args.bsize):
                if tri_emb is not None:
                    if self.args.cl:
                        icl_loss = self.loss(tri_emb, self.train_ill[si:si + bsize])
                    else:
                        icl_loss = self.gcn_nca_loss(tri_emb, self.train_ill[si:si + bsize], [], device=self.device)
                else:
                    icl_loss = self.loss(joint_emb, self.train_ill[si:si + bsize])
                # print('self.device',self.device)
                loss_all = self.gcn_nca_loss(joint_emb, self.train_ill[si:si + bsize], [], device=self.device)

                loss_all = loss_all + icl_loss
                loss_sum_all = loss_sum_all + loss_all

            loss_sum_all.backward()
            self.optimizer.step()

            print("[epoch {:d}] loss_all: {:f}, time: {:.4f} s".format(epoch, loss_sum_all.item(), time.time() - t_epoch))
            del gph_emb, rel_emb, att_emb, joint_emb, tri_emb, ent_rel_emb, name_emb

            if epoch >= self.args.il_start and (epoch + 1) % self.args.semi_learn_step == 0 and self.args.il:
                pred_left, pred_right = self.semi_supervised_learning(epoch)

                if (epoch + 1) % (self.args.semi_learn_step * 10) == self.args.semi_learn_step:
                    new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(pred_left)
                                 if pred_right[p] == i]
                else:
                    new_links = [(self.left_non_train[i], self.right_non_train[p]) for i, p in enumerate(pred_left)
                                 if (pred_right[p] == i)
                                 and ((self.left_non_train[i], self.right_non_train[p]) in new_links)]

            if epoch >= self.args.il_start and (epoch + 1) % (self.args.semi_learn_step * 10) == 0 and len(
                    new_links) != 0 and self.args.il:
                new_links_elect = new_links
                self.train_ill = np.vstack((self.train_ill, np.array(new_links_elect)))
                num_true = len([nl for nl in new_links_elect if nl in self.test_ill_])
                for nl in new_links_elect:
                    self.left_non_train.remove(nl[0])
                    self.right_non_train.remove(nl[1])

                new_links = []

            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Test
            if epoch < self.args.il_start and (epoch + 1) % self.args.check_point == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                max_hit1, max_epoch = self.test(epoch, max_hit1)
                if max_epoch > self.cur_max_epoch:
                    self.cur_max_epoch = max_epoch
            if epoch >= self.args.il_start and (epoch + 1) % self.args.check_point_il == 0:
                print("\n[epoch {:d}] checkpoint!".format(epoch))
                max_hit1, max_epoch = self.test(epoch, max_hit1)
                if max_epoch > self.cur_max_epoch:
                    self.cur_max_epoch = max_epoch
            if self.args.cuda and torch.cuda.is_available():
                torch.cuda.empty_cache()

        print("max_hit1 = {}, cur_max_epoch = {}".format(max_hit1, self.cur_max_epoch))
        print("[optimization finished!]")

        print("[save the EA model finished!]")


    # Test
    def test(self, epoch, max_hit1):
        print("\n[epoch {:d}] checkpoint!".format(epoch))  # 输出当前的训练周期信息
        with torch.no_grad():  # 在无梯度计算的上下文中执行
            t_test = time.time()  # 记录测试开始时间
            self.multiview_encoder.eval()  # 设置模型为评估模式

            # 通过多视图编码器获取嵌入z
            gph_emb, ent_rel_emb, rel_emb, att_emb, joint_emb, tri_emb, name_emb = self.multiview_encoder(
                self.e_input_idx, self.r_input_idx, self.e_in, self.e_out,
                self.e_adj, epoch, self.r_in_adj, self.r_out_adj,
                self.r_path_adj, self.edge_index_all,
                self.rel_all, self.ent_features,
                self.rel_features_in, self.rel_features_out,
                self.att_features
            )

            # 计算融合权重的归一化
            w_normalized = F.softmax(self.multiview_encoder.fusion.weight, dim=0)
            print("normalised weights:", w_normalized.data.squeeze())  # 输出归一化后的权重
            if tri_emb is not None:
                tri_emb = F.normalize(tri_emb)  # 对三元组嵌入进行归一化

            joint_emb = F.normalize(joint_emb)  # 对联合嵌入进行归一化

            # 定义 top_k 评估指标
            top_k = [1, 5, 10, 50]
            if "100" in self.args.file_dir:
                pass  # 如果文件目录中包含 "100"，不进行特定处理
            else:
                acc_l2r = np.zeros((len(top_k)), dtype=np.float32)  # 左到右的准确率数组
                acc_r2l = np.zeros((len(top_k)), dtype=np.float32)  # 右到左的准确率数组
                test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.  # 初始化统计变量

                # 根据距离计算方式选择距离计算的方法
                if self.args.dist == 2:
                    distance_joi = pairwise_distances(joint_emb[self.test_left],
                                                      joint_emb[self.test_right])  # 计算联合嵌入的距离
                    if tri_emb is not None:
                        distance_tri = pairwise_distances(tri_emb[self.test_left],
                                                          tri_emb[self.test_right])  # 计算三元组嵌入的距离
                        distance_f = distance_joi + distance_tri  # 综合距离
                    else:
                        distance_f = distance_joi  # 仅使用联合嵌入的距离
                elif self.args.dist == 1:
                    # 使用城市街区距离计算距离
                    distance_joi = torch.FloatTensor(spatial.distance.cdist(
                        joint_emb[self.test_left].cpu().data.numpy(),
                        joint_emb[self.test_right].cpu().data.numpy(), metric="cityblock"))
                    if tri_emb is not None:
                        distance_tri = torch.FloatTensor(spatial.distance.cdist(
                            tri_emb[self.test_left].cpu().data.numpy(),
                            tri_emb[self.test_right].cpu().data.numpy(), metric="cityblock"))
                        distance_f = distance_joi + distance_tri  # 综合距离
                    else:
                        distance_f = distance_joi  # 仅使用联合嵌入的距离
                else:
                    raise NotImplementedError  # 如果距离类型未实现，则抛出错误
                distance = distance_f  # 设置计算得到的距离

                # 如果使用 CSLS (Cross-Domain Similarity Learning)
                if self.args.csls is True:
                    distance = 1 - csls_sim(1 - distance, self.args.csls_k)  # 应用 CSLS 相似度调整

                # 在最后一个周期保存预测结果
                if epoch + 1 == self.args.epochs:
                    to_write = []  # 初始化写入数据列表
                    test_left_np = self.test_left.cpu().numpy()  # 将测试左侧的 Tensor 转换为 NumPy 数组
                    test_right_np = self.test_right.cpu().numpy()  # 将测试右侧的 Tensor 转换为 NumPy 数组
                    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])  # 写入表头

                # 计算从左到右的排名和准确率
                for idx in range(self.test_left.shape[0]):
                    values, indices = torch.sort(distance[idx, :], descending=False)  # 对距离进行排序
                    rank = (indices == idx).nonzero().squeeze().item()  # 找到当前样本的排名

                    mean_l2r += (rank + 1)  # 累加排名
                    mrr_l2r += 1.0 / (rank + 1)  # 累加倒数排名
                    for i in range(len(top_k)):  # 计算各个 top_k 的准确率
                        if rank < top_k[i]:
                            acc_l2r[i] += 1
                    if epoch + 1 == self.args.epochs:  # 在最后一个周期写入预测结果
                        indices = indices.cpu().numpy()  # 将索引转换为 NumPy 数组
                        to_write.append(
                            [idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
                             test_right_np[indices[1]], test_right_np[indices[2]]])
                if epoch + 1 == self.args.epochs:
                    import csv  # 导入 CSV 模块
                    save_path = self.args.save_path  # 获取保存路径
                    if not os.path.exists(save_path):
                        os.mkdir(save_path)  # 如果目录不存在，创建目录
                    with open(os.path.join(save_path, "pred.txt"), "w") as f:  # 打开文件以写入预测结果
                        wr = csv.writer(f, dialect='excel')  # 创建 CSV 写入器
                        wr.writerows(to_write)  # 写入数据

                # 计算从右到左的排名和准确率
                for idx in range(self.test_right.shape[0]):
                    _, indices = torch.sort(distance[:, idx], descending=False)  # 对每个右侧样本计算距离排序
                    rank = (indices == idx).nonzero().squeeze().item()  # 找到当前样本的排名
                    mean_r2l += (rank + 1)  # 累加排名
                    mrr_r2l += 1.0 / (rank + 1)  # 累加倒数排名
                    for i in range(len(top_k)):  # 计算各个 top_k 的准确率
                        if rank < top_k[i]:
                            acc_r2l[i] += 1

                # 计算平均排名和 MRR
                mean_l2r /= self.test_left.size(0)
                mean_r2l /= self.test_right.size(0)
                mrr_l2r /= self.test_left.size(0)
                mrr_r2l /= self.test_right.size(0)
                for i in range(len(top_k)):
                    acc_l2r[i] = round(acc_l2r[i] / self.test_left.size(0), 4)  # 计算从左到右的准确率
                    acc_r2l[i] = round(acc_r2l[i] / self.test_right.size(0), 4)  # 计算从右到左的准确率

                def visualize_embeddings(embeddings, labels, title="Entity Embeddings"):
                    # 使用PCA或t-SNE进行降维
                    pca = PCA(n_components=2)
                    tsne = TSNE(n_components=2)
                    embeddings_pca = pca.fit_transform(embeddings)
                    embeddings_tsne = tsne.fit_transform(embeddings)

                    # 绘制降维后的嵌入
                    plt.figure(figsize=(10, 8))
                    for label in np.unique(labels):
                        indices = np.where(labels == label)
                        plt.scatter(embeddings_pca[indices, 0], embeddings_pca[indices, 1], label=label)
                    plt.title(title)
                    plt.legend()
                    plt.show()

                # 在test函数中调用可视化函数
                if epoch + 1 == self.args.epochs:
                    print('visualize_embeddings')
                    visualize_embeddings(joint_emb.detach().cpu().numpy(), self.test_left.cpu().numpy(),
                                         "Joint Embeddings at Epoch {}".format(epoch))
                # 清理不再需要的变量以释放内存
                del gph_emb, rel_emb, att_emb, joint_emb, tri_emb, ent_rel_emb, name_emb
                gc.collect()  # 垃圾回收

            avg_hit = (acc_l2r + acc_r2l) / 2  # 计算平均命中率
            # 输出左到右和右到左的准确率、平均排名和 MRR
            print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                                mean_l2r, mrr_l2r,
                                                                                                time.time() - t_test))
            print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_r2l,
                                                                                                mean_r2l, mrr_r2l,
                                                                                                time.time() - t_test))
            print("avg_hit = {}\tavg_mr = {:.3f}\tavg_mrr={:.3f}\n".format(avg_hit, (mean_l2r + mean_r2l) / 2,
                                                                           (mrr_l2r + mrr_r2l) / 2))

            # 如果平均命中率超过最大值，更新最大值和当前周期


            if avg_hit[0] > max_hit1:
                save_path = os.path.join(self.args.save_path, "model_myEA.pth")
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

                torch.save(self.multiview_encoder.state_dict(), save_path)
                # torch.save(self.multiview_encoder, os.path.join(self.args.save_path, "model_myEA.pth"))
                return avg_hit[0], epoch
            else:
                return max_hit1, self.cur_max_epoch


if __name__ == "__main__":
    model = SCMEA()
    model.train()
