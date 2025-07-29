import json  # 导入 JSON 处理模块
import pickle  # 导入 pickle 模块，用于序列化和反序列化
from collections import Counter  # 从 collections 模块导入 Counter，用于计数

import numpy as np  # 导入 NumPy 库
from tqdm import tqdm  # 导入 tqdm 库，用于进度条显示

# 加载文件的函数，返回指定数量的整数元组
def loadfile(fn, num=1):
    print('loading a file...' + fn)  # 输出正在加载的文件名
    ret = []  # 初始化返回列表
    with open(fn, encoding='utf-8') as f:  # 打开文件
        for line in f:  # 遍历文件中的每一行
            th = line[:-1].split('\t')  # 按制表符分割行，并去掉末尾换行符
            x = []  # 初始化一个列表用于存储整数
            for i in range(num):  # 遍历指定数量
                x.append(int(th[i]))  # 将分割的字符串转换为整数并添加到列表
            ret.append(tuple(x))  # 将列表转换为元组并添加到返回列表中
    return ret  # 返回元组列表

# 获取 ID 列表的函数
def get_ids(fn):
    ids = []  # 初始化 ID 列表
    with open(fn, encoding='utf-8') as f:  # 打开文件
        for line in f:  # 遍历文件中的每一行
            th = line[:-1].split('\t')  # 按制表符分割行，并去掉末尾换行符
            ids.append(int(th[0]))  # 将第一列的字符串转换为整数并添加到 ID 列表中
    return ids  # 返回 ID 列表

# 获取实体到 ID 映射的函数
def get_ent2id(fns):
    ent2id = {}  # 初始化实体到 ID 的字典
    for fn in fns:  # 遍历输入的文件名列表
        with open(fn, 'r', encoding='utf-8') as f:  # 打开每个文件
            for line in f:  # 遍历文件中的每一行
                th = line[:-1].split('\t')  # 按制表符分割行，并去掉末尾换行符
                ent2id[th[1]] = int(th[0])  # 将实体名称作为键，ID 作为值存入字典
    return ent2id  # 返回实体到 ID 的映射字典

# 加载属性的函数
def load_attr(fns, e, ent2id, topA=1000):
    cnt = {}  # 初始化计数字典
    # fns:[attrs_1, attrs_2]
    for fn in fns:  # 遍历输入的属性文件
        with open(fn, 'r', encoding='utf-8') as f:  # 打开每个文件
            for line in f:  # 遍历文件中的每一行
                th = line[:-1].split('\t')  # 按制表符分割行，并去掉末尾换行符
                if th[0] not in ent2id:  # 如果实体不在 ID 映射中，跳过
                    continue
                for i in range(1, len(th)):  # 遍历属性列
                    if th[i] not in cnt:  # 如果属性不在计数字典中
                        cnt[th[i]] = 1  # 初始化计数为 1
                    else:
                        cnt[th[i]] += 1  # 计数加 1
    fre = [(k, cnt[k]) for k in sorted(cnt, key=cnt.get, reverse=True)]  # 按计数排序属性
    print("len(fre",len(fre))
    if len(fre) < 1000:  # 如果属性数量少于 1000
        topA = len(fre)  # 更新 topA 为实际属性数量
    attr2id = {}  # 初始化属性到 ID 的映射字典
    for i in range(topA):  # 选择前 topA 个属性
        attr2id[fre[i][0]] = i  # 将属性和对应的 ID 存入字典
    attr = np.zeros((e, topA), dtype=np.float32)  # 初始化属性矩阵

    for fn in fns:  # 遍历输入的属性文件
        with open(fn, 'r', encoding='utf-8') as f:  # 打开每个文件
            for line in f:  # 遍历文件中的每一行
                th = line[:-1].split('\t')  # 按制表符分割行，并去掉末尾换行符
                if th[0] in ent2id:  # 如果实体在 ID 映射中
                    for i in range(1, len(th)):  # 遍历属性列
                        if th[i] in attr2id:  # 如果属性在 attr2id 中
                            attr[ent2id[th[0]]][attr2id[th[i]]] = 1.0  # 设置属性矩阵对应位置为 1
    return attr  # 返回属性矩阵

# 加载前 topR 个关系的函数
def load_top_relation(e, triples, topR=1000):
    rel_mat = np.zeros((e, topR), dtype=np.float32)  # 初始化关系矩阵
    rels = np.array(triples)[:, 1]  # 提取所有关系
    top_rels = Counter(rels).most_common(topR)  # 统计并获取前 topR 个关系
    rel_index_dict = {r: i for i, (r, cnt) in enumerate(top_rels)}  # 创建关系索引字典
    for tri in triples:  # 遍历三元组
        h = tri[0]  # 头实体
        r = tri[1]  # 关系
        o = tri[2]  # 尾实体
        if r in rel_index_dict:  # 如果关系在索引字典中
            rel_mat[h][rel_index_dict[r]] += 1.  # 增加头实体的关系计数
            rel_mat[o][rel_index_dict[r]] += 1.  # 增加尾实体的关系计数
    return np.array(rel_mat)  # 返回关系矩阵

# 加载关系的函数
def load_relation(e_num, r_num, triples, topR=1000):
    rel_mat_in = np.zeros((e_num, r_num), dtype=np.float32)  # 初始化输入关系矩阵
    rel_mat_out = np.zeros((e_num, r_num), dtype=np.float32)  # 初始化输出关系矩阵
    for (h, r, t) in triples:  # 遍历三元组
        rel_mat_in[t][r] += 1  # 更新尾实体的输入关系计数
        rel_mat_out[h][r] += 1  # 更新头实体的输出关系计数
    return np.array(rel_mat_in), np.array(rel_mat_out)  # 返回输入和输出关系矩阵

# 加载词向量的函数
def load_word_emb(filename):
    """
        请从 "http://nlp.stanford.edu/data/glove.6B.zip" 下载压缩文件
        并选择 "glove.6B.300d.txt" 作为词向量。
    """
    word_vecs = {}  # 初始化词向量字典
    print("load word_emb......")  # 输出加载提示
    with open(filename, encoding='UTF-8') as f:  # 打开词向量文件
        for line in tqdm(f.readlines()):  # 遍历文件中的每一行并显示进度
            line = line.split()  # 按空格分割行
            word_vecs[line[0]] = np.array([float(x) for x in line[1:]])  # 将词和对应的向量存入字典
    return word_vecs  # 返回词向量字典

# 加载翻译后的实体名称的函数
def load_trans_ent_name(file_path):
    # 翻译后的实体名称
    ent_names = json.load(open(file_path, "r"))  # 从 JSON 文件加载实体名称
    return ent_names  # 返回实体名称字典

# 加载单词和字符特征的函数
def word_char_f(node_size, ent_names, word_vecs):
    d = {}  # 初始化二元字符字典
    count = 0  # 字符计数器
    for _, name in ent_names:  # 遍历实体名称
        for word in name:  # 遍历每个单词
            word = word.lower()  # 将单词转换为小写
            for idx in range(len(word) - 1):  # 遍历单词中的每个字符对
                if word[idx:idx + 2] not in d:  # 如果字符对不在字典中
                    d[word[idx:idx + 2]] = count  # 添加字符对及其索引
                    count += 1  # 计数器加 1

    ent_vec = np.zeros((node_size, 300))  # 初始化实体向量矩阵
    char_vec = np.zeros((node_size, len(d)))  # 初始化字符向量矩阵

    for i, name in ent_names:  # 遍历实体名称
        k = 0  # 有效单词计数器
        for word in name:  # 遍历每个单词
            word = word.lower()  # 将单词转换为小写
            if word in word_vecs:  # 如果单词在词向量字典中
                ent_vec[i] += word_vecs[word]  # 累加对应的词向量
                k += 1  # 有效单词计数加 1

            for idx in range(len(word) - 1):  # 遍历单词中的每个字符对
                char_vec[i, d[word[idx:idx + 2]]] += 1  # 统计字符对出现次数
        if k:  # 如果有有效单词
            ent_vec[i] /= k  # 对实体向量进行平均
        else:
            ent_vec[i] = np.random.random(300) - 0.5  # 如果没有有效单词，随机初始化

        if np.sum(char_vec[i]) == 0:  # 如果字符向量全部为 0
            char_vec[i] = np.random.random(len(d)) - 0.5  # 随机初始化
        ent_vec[i] = ent_vec[i] / np.linalg.norm(ent_vec[i])  # 归一化实体向量
        char_vec[i] = char_vec[i] / np.linalg.norm(char_vec[i])  # 归一化字符向量

    return ent_vec, char_vec  # 返回实体向量和字符向量

# 加载 JSON 格式的嵌入数据的函数
def load_json_embd(path):
    embd_dict = {}  # 初始化嵌入字典
    with open(path) as f:  # 打开 JSON 文件
        for line in f:  # 遍历文件中的每一行
            example = json.loads(line.strip())  # 解析 JSON 数据
            vec = np.array([float(e) for e in example['feature'].split()])  # 获取嵌入向量
            embd_dict[int(example['guid'])] = vec  # 将 GUID 和对应的嵌入向量存入字典
    return embd_dict  # 返回嵌入字典
