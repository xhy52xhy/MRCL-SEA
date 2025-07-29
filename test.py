#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import csv
from tqdm import tqdm
import os
from torch_geometric.utils import sort_edge_index
from lion_optim import *
from models import *
from utils import *
from loss import *
from load import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def visualization_analysis(joint_emb, tri_emb, test_left, test_right, ent2id_dict, save_path):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import seaborn as sns



    joint_emb = joint_emb.cpu()
    tri_emb = tri_emb.cpu()
    test_left = test_left.cpu()
    test_right = test_right.cpu()

    joint_emb = F.normalize(joint_emb)
    if tri_emb is not None:
        tri_emb = F.normalize(tri_emb)



    np.random.seed(2024)

    num_pairs_list = [200, 500, 1000]
    titles = ['(c) 200 Entity Distribution',
              '(d) 500 Entity Distribution',
              '(e) 1000 Entity Distribution']


    plt.figure(figsize=(18, 6))  

    for idx, num_pairs in enumerate(num_pairs_list):
      
        pair_indices = np.random.choice(len(test_left), num_pairs, replace=False)
        selected_left = test_left[pair_indices]
        selected_right = test_right[pair_indices]


        d_joi = pairwise_distances(joint_emb[selected_left], joint_emb[selected_right])
        if tri_emb is not None:
            d_tri = pairwise_distances(tri_emb[selected_left], tri_emb[selected_right])
            distance_matrix = d_joi + d_tri
        else:
            distance_matrix = d_joi

        similarity_matrix = 1 / (1 + distance_matrix.cpu().numpy())

        if tri_emb is not None:
            print(joint_emb.size(), tri_emb.size())
            final_emb = joint_emb
        else:
            final_emb = joint_emb

        final_emb = final_emb.numpy()


        left_emb = final_emb[selected_left]
        right_emb = final_emb[selected_right]

        pca = PCA(n_components=2)
        all_emb = np.vstack([left_emb, right_emb])
        all_emb_2d = pca.fit_transform(all_emb)
        left_2d = all_emb_2d[:num_pairs]
        right_2d = all_emb_2d[num_pairs:]


        plt.subplot(1, 3, idx + 1)  
        plt.scatter(left_2d[:, 0], left_2d[:, 1], c='g', label='Source KG', alpha=0.7)
        plt.scatter(right_2d[:, 0], right_2d[:, 1], c='b', label='Target KG', alpha=0.7)

        for i in range(num_pairs):
            plt.plot([left_2d[i, 0], right_2d[i, 0]],
                     [left_2d[i, 1], right_2d[i, 1]],
                     'gray', linestyle='--', alpha=0.3)

        plt.legend()
        plt.title(titles[idx]) 

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'combined_visualization.png'), dpi=300)
    plt.close()

  
    def get_entity_name(ent_id, ent2id_dict):
        for name, idx in ent2id_dict.items():
            if idx == ent_id:
                return name
        return str(ent_id)


    print("\nCase Study - Example Alignments:")
    print("Source Entity -> Target Entity (Distance Score)")
    print("-" * 60)
    for i in range(50):  
        src_id = selected_left[i].item()
        tgt_id = selected_right[i].item()
        src_name = get_entity_name(src_id, ent2id_dict)
        tgt_name = get_entity_name(tgt_id, ent2id_dict)
        distance_score = distance_matrix[i][i].item()
        print(f"{src_name} -> {tgt_name} (Distance: {distance_score:.3f})")



class TestModel:
    def __init__(self, dataset):
        self.ent_name_dim = None
        self.all_duplicates=None
        self.ent_features = None
        self.triples_2 = None
        self.triples_1 = None
        self.ent2id_dict = None
        self.ills = None
        self.triples = None
        self.ids = None
        self.left_ents = None
        self.right_ents = None
        self.edge_index_all = None
        self.rel_all = None
        self.rel_features_in = None
        self.rel_features_out = None
        self.att_features = None
        self.ENT_NUM = None
        self.REL_NUM = None
        self.e_adj = None
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
        self.e_input_idx = None
        self.r_input_idx = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.file_dir = dataset
        self.set_seed(2023, cuda=True if torch.cuda.is_available() else "cpu")

        self.init_data()
        self.init_emb()

        self.e_input_idx = torch.LongTensor(np.arange(self.ENT_NUM)).to(self.device)
        self.r_input_idx = torch.LongTensor(np.arange(self.REL_NUM)).to(self.device)

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
        file_dir = self.file_dir
        device = self.device

        self.ent2id_dict, self.all_duplicates,self.ills, self.triples_1, self.triples_2, self.triples, self.ids, self.rel_ht_dict = read_raw_data(
            file_dir, kg_list)
        e1 = os.path.join(file_dir, 'ent_ids_1')
        e2 = os.path.join(file_dir, 'ent_ids_2')
        self.left_ents = get_ids(e1)
        self.right_ents = get_ids(e2)
        self.ENT_NUM = len(self.ent2id_dict)+len(self.all_duplicates)
        self.REL_NUM = len(self.rel_ht_dict)
        self.test_ill_ = self.ills[int(len(self.ills) // 1 * 0.3):]
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

    def init_emb(self):
        file_dir = self.file_dir
        device = self.device
        sr_ent_embed, tg_ent_embed = np.load('data/DBP15K/zh_en/sr_features_bge.npy'), np.load(
            'data/DBP15K/zh_en/tg_features_bge.npy')
        sr_ent_embed, tg_ent_embed = torch.tensor(sr_ent_embed).to(self.device), torch.tensor(tg_ent_embed).to(
            self.device)
        self.ent_features = torch.cat((sr_ent_embed, tg_ent_embed), dim=0)
        # self.ent_features = torch.Tensor(ent_features).to(device)
        self.ent_features.requires_grad = True
        self.ent_name_dim = self.ent_features.shape[1]
        a1 = os.path.join(file_dir, 'training_attrs_1')
        a2 = os.path.join(file_dir, 'training_attrs_2')
        print(self.ENT_NUM,len(self.ent2id_dict))
        self.att_features = load_attr([a1, a2], self.ENT_NUM, self.ent2id_dict, 1000)
        self.att_features = torch.Tensor(self.att_features).to(device)
        self.rel_features_in, self.rel_features_out = load_relation(self.ENT_NUM, self.REL_NUM, self.triples)
        self.rel_features_in = torch.Tensor(self.rel_features_in).to(device)
        self.rel_features_out = torch.Tensor(self.rel_features_out).to(device)
        self.e_adj = get_adjr(self.ENT_NUM, self.triples, norm=True)  # getting a sparse tensor r_adj
        self.e_adj = self.e_adj.to(self.device)


dataset = "data/DBP15K/zh_en"
sub_kg = dataset.split('/')[-1]

model_path = os.path.join("../save"+"/model_myEA.pth")
print(model_path)
net = TestModel(dataset)

net.multiview_encoder = torch.load(model_path,
                                   map_location=torch.device('cuda') if torch.cuda.is_available() else "cpu")

save_path = os.path.join("../test_logs/", sub_kg)

if not os.path.exists(save_path):
    os.mkdir(save_path)
'''
with torch.no_grad():
    print("[Start testing...] ")
    t_test = time.time()
    net.multiview_encoder.eval()
    _, _, _, _, joint_emb, tri_emb, _ = net.multiview_encoder(net.e_input_idx, net.r_input_idx, net.e_in, net.e_out,
                                                              net.e_adj, -1, net.r_in_adj, net.r_out_adj,
                                                              net.r_path_adj, net.edge_index_all, net.rel_all,
                                                              net.ent_features,
                                                              net.rel_features_in, net.rel_features_out,
                                                              net.att_features)

    tri_emb = F.normalize(tri_emb)
    joint_emb = F.normalize(joint_emb)

    top_k = [1, 5, 10, 50]
    acc_l2r = np.zeros((len(top_k)), dtype=np.float32)
    acc_r2l = np.zeros((len(top_k)), dtype=np.float32)
    test_total, test_loss, mean_l2r, mean_r2l, mrr_l2r, mrr_r2l = 0, 0., 0., 0., 0., 0.
    distance_joi = pairwise_distances(joint_emb[net.test_left], joint_emb[net.test_right])
    distance_tri = pairwise_distances(tri_emb[net.test_left], tri_emb[net.test_right])
    distance_f = distance_joi + distance_tri
    distance = 1 - csls_sim(1 - distance_f, 10)

    to_write = []
    test_left_np = net.test_left.cpu().numpy()
    test_right_np = net.test_right.cpu().numpy()
    to_write.append(["idx", "rank", "query_id", "gt_id", "ret1", "ret2", "ret3"])

    print("[Testing from left to right...] ")
    for idx in tqdm(range(net.test_left.shape[0])):
        values, indices = torch.sort(distance[idx, :], descending=False)
        rank = (indices == idx).nonzero().squeeze().item()

        mean_l2r += (rank + 1)
        mrr_l2r += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_l2r[i] += 1
        indices = indices.cpu().numpy()
        to_write.append(
            [idx, rank, test_left_np[idx], test_right_np[idx], test_right_np[indices[0]],
             test_right_np[indices[1]], test_right_np[indices[2]]])

    with open(os.path.join(save_path, "pred.txt"), "w") as f:
        wr = csv.writer(f, dialect='excel')
        wr.writerows(to_write)

    print("[Testing from right to left...] ")
    for idx in tqdm(range(net.test_right.shape[0])):
        _, indices = torch.sort(distance[:, idx], descending=False)
        rank = (indices == idx).nonzero().squeeze().item()
        mean_r2l += (rank + 1)
        mrr_r2l += 1.0 / (rank + 1)
        for i in range(len(top_k)):
            if rank < top_k[i]:
                acc_r2l[i] += 1

    mean_l2r /= net.test_left.size(0)
    mean_r2l /= net.test_right.size(0)
    mrr_l2r /= net.test_left.size(0)
    mrr_r2l /= net.test_right.size(0)
    for i in range(len(top_k)):
        acc_l2r[i] = round(acc_l2r[i] / net.test_left.size(0), 4)
        acc_r2l[i] = round(acc_r2l[i] / net.test_right.size(0), 4)

    avg_hit = (acc_l2r + acc_r2l) / 2
    print("l2r: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_l2r,
                                                                                        mean_l2r, mrr_l2r,
                                                                                        time.time() - t_test))
    print("r2l: acc of top {} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.4f} s ".format(top_k, acc_r2l,
                                                                                        mean_r2l, mrr_r2l,
                                                                                        time.time() - t_test))
    print("avg_hit = {}\tavg_mr = {:.3f}\tavg_mrr={:.3f}\n".format(avg_hit, (mean_l2r + mean_r2l) / 2,
                                                                   (mrr_l2r + mrr_r2l) / 2))
    # Add after the testing results
    print("\n[Generating visualizations and case study...]")
    visualization_analysis(joint_emb, tri_emb,
                           net.test_left, net.test_right,
                           net.ent2id_dict, save_path)
'''





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_triples(file_path):
    # 加载三元组数据
    triples = pd.read_csv(file_path, sep='\t', header=None)
    return triples.values
triples_1 = load_triples('../data/DBP15K/zh_en/triples_1')
triples_2 = load_triples('../data/DBP15K/zh_en/triples_2')
def get_neighbor_distribution(triples):
    # 计算一跳邻居
    one_hop_neighbors = defaultdict(set)
    for h, r, t in triples:
        one_hop_neighbors[h].add(t)
        one_hop_neighbors[t].add(h)

    # 计算二跳邻居
    two_hop_neighbors = defaultdict(set)
    for node in one_hop_neighbors:
        for neighbor in one_hop_neighbors[node]:
            two_hop_neighbors[node].update(one_hop_neighbors[neighbor])
            two_hop_neighbors[node].discard(node)

    # 统计分布
    distribution = defaultdict(list)
    for node in one_hop_neighbors:
        one_hop_count = len(one_hop_neighbors[node])
        two_hop_count = len(two_hop_neighbors[node])
        distribution[one_hop_count].append(two_hop_count)

    return distribution


def get_entity_neighbor_count(triples, ent_num):
    neighbor_counts = defaultdict(int)
    for h, _, t in triples:
        neighbor_counts[h] += 1
        neighbor_counts[t] += 1
    return neighbor_counts


def plot_hits_distribution(results, save_path):
    plt.figure(figsize=(12, 6))

    ranges = list(results.keys())
    x = np.arange(len(ranges))
    width = 0.2

    # 使用正确的数据绘制柱状图
    hit1_values = [results[r]['hits'][0] for r in ranges]
    hit5_values = [results[r]['hits'][1] for r in ranges]
    hit10_values = [results[r]['hits'][2] for r in ranges]
    hit50_values = [results[r]['hits'][3] for r in ranges]

    # 绘制柱状图
    bar1 = plt.bar(x - width * 1.5, hit1_values, width, label='Hit@1', color='#f1ddbf')
    bar2 = plt.bar(x - width / 2, hit5_values, width, label='Hit@5', color='#525e75')
    bar3 = plt.bar(x + width / 2, hit10_values, width, label='Hit@10', color='#78938a')
    bar4 = plt.bar(x + width * 1.5, hit50_values, width, label='Hit@50', color='#92ba92')

    # 添加数值标签
    import math

    for bar in bar1:
        yval = bar.get_height()
        truncated_val = math.floor(yval * 100) / 100  # Truncate to two decimal places
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{truncated_val:.2f}', ha='center', va='bottom',
                 fontsize=16)

    for bar in bar2:
        yval = bar.get_height()
        truncated_val = math.floor(yval * 100) / 100  # Truncate to two decimal places
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{truncated_val:.2f}', ha='center', va='bottom',
                 fontsize=16)

    for bar in bar3:
        yval = bar.get_height()
        truncated_val = math.floor(yval * 100) / 100  # Truncate to two decimal places
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{truncated_val:.2f}', ha='center', va='bottom',
                 fontsize=16)

    for bar in bar4:
        yval = bar.get_height()
        truncated_val = math.floor(yval * 100) / 100  # Truncate to two decimal places
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{truncated_val:.2f}', ha='center', va='bottom',
                 fontsize=16)

    # 设置标签和标题
    plt.xlabel('Neighbor Count Range', fontsize=24)
    plt.ylabel('Hit', fontsize=26)
    plt.title('(d) Entity Alignment Performance by Neighbor Count', fontsize=24)

    # 设置x轴和y轴的刻度字体大小
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    plt.xticks(x, ranges)
    plt.legend(fontsize=20)

    # 设置y轴范围从0到1
    plt.ylim(0, 1.1)

    # 添加网格线以提高可读性
    plt.grid(True, axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'hits_by_neighbors.png'))
    plt.close()
def calculate_hits_by_neighbor_range(test_left, test_right, distance, neighbor_counts, ranges):
    results = {range_name: {'indices': [], 'hits': np.zeros(4)} for range_name in ranges.keys()}
    top_k = [1, 5, 10, 50]

    # 首先将测试实体按邻居数量分类
    for idx, left_ent in enumerate(test_left):
        neighbor_count = neighbor_counts[left_ent]
        for range_name, (min_n, max_n) in ranges.items():
            if min_n <= neighbor_count < max_n:
                results[range_name]['indices'].append(idx)
                break

    # 对每个范围计算hits
    for range_name, data in results.items():
        test_indices = data['indices']
        if not test_indices:
            continue

        for idx in test_indices:
            # 获取当前测试实体的距离向量
            values, indices = torch.sort(distance[idx, :], descending=False)
            # 找到正确匹配的排名
            rank = (indices == idx).nonzero().squeeze().item()

            # 计算hits@k
            for i, k in enumerate(top_k):
                if rank < k:
                    results[range_name]['hits'][i] += 1

        # 计算正确率
        test_size = len(test_indices)
        if test_size > 0:
            results[range_name]['hits'] = results[range_name]['hits'] / test_size

    return results


# 修改测试代码部分
with torch.no_grad():
    print("[Start testing...] ")
    t_test = time.time()
    net.multiview_encoder.eval()

    # 获取实体嵌入
    _, _, _, _, joint_emb, tri_emb, _ = net.multiview_encoder(net.e_input_idx, net.r_input_idx, net.e_in, net.e_out,
                                                              net.e_adj, -1, net.r_in_adj, net.r_out_adj,
                                                              net.r_path_adj, net.edge_index_all, net.rel_all,
                                                              net.ent_features,
                                                              net.rel_features_in, net.rel_features_out,
                                                              net.att_features)

    tri_emb = F.normalize(tri_emb)
    joint_emb = F.normalize(joint_emb)

    # 计算距离矩阵
    distance_joi = pairwise_distances(joint_emb[net.test_left], joint_emb[net.test_right])
    distance_tri = pairwise_distances(tri_emb[net.test_left], tri_emb[net.test_right])
    distance_f = distance_joi + distance_tri
    distance = 1 - csls_sim(1 - distance_f, 10)

    # 计算每个实体的邻居数量
    neighbor_counts = defaultdict(int)
    for h, _, t in net.triples:
        neighbor_counts[h] += 1
        neighbor_counts[t] += 1

    # 定义邻居数量范围
    neighbor_ranges = {
        '0-10': (0, 10),
        '10-20': (10, 20),
        '20-30': (20, 30),
        '30+': (30, float('inf'))
    }

    # 计算不同邻居范围的hits
    results = calculate_hits_by_neighbor_range(
        net.test_left.cpu().numpy(),
        net.test_right.cpu().numpy(),
        distance,
        neighbor_counts,
        neighbor_ranges
    )
    plot_hits_distribution(results, save_path)
    # 打印结果
    print("\nHits by neighbor count range:")
    for range_name, data in results.items():
        print(f"\n{range_name} neighbors:")
        print(f"Number of entities: {len(data['indices'])}")
        print(f"Hit@1: {data['hits'][0]:.4f}")
        print(f"Hit@5: {data['hits'][1]:.4f}")
        print(f"Hit@10: {data['hits'][2]:.4f}")
        print(f"Hit@50: {data['hits'][3]:.4f}")

