# # 定义输入和输出文件名
# input_filename = 'entity2id_en.txt'
# output_filename = 'ent_ids_1'
#
# # 打开原始文件进行读取，同时打开一个新文件用于写入修改后的内容
# with open(input_filename, 'r', encoding='utf-8') as infile, \
#      open(output_filename, 'w', encoding='utf-8') as outfile:
#     # 逐行读取文件
#     for line in infile:
#         # 移除行尾的换行符并以\t分割行
#         parts = line.strip().split('\t')
#         # 检查行是否确实有两列
#         if len(parts) == 2:
#             # 交换两列的位置
#             swapped_parts = parts[1] + '\t' + parts[0]
#             # 将交换后的内容写入新文件
#             outfile.write(swapped_parts + '\n')
#         else:
#             # 如果行不包含恰好两列，可以打印错误或采取其他操作
#             print(f"Warning: Line does not contain exactly two columns: {line}")

# 定义输入和输出文件名
# def find_missing_values(input_filename):
#     # 读取文件并提取第二列的值
#     with open(input_filename, 'r', encoding='utf-8') as file:
#         columns = []
#         for line in file:
#             parts = line.strip().split('\t')
#             if len(parts) >= 2:
#                 try:
#                     columns.append(int(parts[1]))
#                 except ValueError:
#                     print(f"Warning: Non-integer value found in the second column: {parts[1]}")
#
#     # 检查是否有缺失的值
#     missing_values = set()
#     expected_value = 1
#     for value in sorted(columns):
#         while expected_value < value:
#             missing_values.add(expected_value)
#             expected_value += 1
#         expected_value = max(expected_value, value + 1)
#
#     return missing_values
#
#
# # 定义输入文件名
# input_filename = 'triples_1'
#
# # 找到并打印缺失的值
# missing_values = find_missing_values(input_filename)
# print("Missing values in the second column:")
# for value in sorted(missing_values):
#     print(value)

# 读取原始文件
# with open('triples_2', 'r', encoding='utf-8') as file:
#     lines = file.readlines()
#
# # 提取第二列编号
# original_numbers = [line.split('\t')[1].strip() for line in lines]
#
# # 去重并排序
# unique_numbers = sorted(set(original_numbers), key=int)
#
# # 生成等差数列编号映射
# number_mapping = {num: str(i) for i, num in enumerate(unique_numbers)}
#
# # 替换原编号为等差数列
# new_lines = []
# for line in lines:
#     parts = line.split('\t')
#     parts[1] = number_mapping[parts[1].strip()]  # 替换第二列编号
#     new_lines.append('\t'.join(parts))
#
# # 写回文件
# with open('triples_2_updated', 'w', encoding='utf-8') as file:
#     file.writelines(new_lines)
#
# print("编号已重新排列并保存到文件 'triples_1_updated.txt'.")

# 打开原始文件和结果文件
# # 打开原始文件和结果文件，指定编码为UTF-8
# with open('triples_2', 'r', encoding='utf-8') as file, open('triples_2_mo', 'w',
#                                                                 encoding='utf-8') as output_file:
#     # 遍历文件的每一行
#     for line in file:
#         # 使用制表符分割行
#         columns = line.strip().split('\t')
#
#         # 将第一列的值转换为整数并加上24944
#         columns[0] = str(int(columns[0]) + 24944)
#         columns[2] = str(int(columns[2]) + 24944)
#         # 将修改后的第一列与其他列重新组合，并写入结果文件
#         modified_line = '\t'.join(columns) + '\n'
#         output_file.write(modified_line)
# 假设你的文件名为atts_properties_en.txt
input_file = 'data/DBP15K/zh_en/att-properties_zh.txt'
output_file = 'data/DBP15K/zh_en/training_attrs_2'

# 用于存储化合物与其所有属性的映射
compound_attrs = {}

with open(input_file, 'r',encoding='utf-8') as infile:
    for line in infile:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            compound, attr_name,_ = parts
            if compound not in compound_attrs:
                compound_attrs[compound] = set()
            compound_attrs[compound].add(attr_name)

# 将属性集合转换为排序后的列表，并写入文件
with open(output_file, 'w',encoding='utf-8') as outfile:
    for compound, attrs in compound_attrs.items():
        attr_list = sorted(list(attrs))
        attr_string = '\t'.join(attr_list)
        outfile.write(f'{compound}\t{attr_string}\n')

