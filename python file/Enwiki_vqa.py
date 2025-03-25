import pandas as pd
import ijson
import json

# 文件路径设置
train_csv_path = r"E:\train.csv"
test_csv_path = r"E:\test.csv"
kb_wiki_path = r"E:\encyclopedic_kb_wiki.json"
output_path = r"E:\encyclopedic_kb_wiki_vqa.json"

# 1. 读取 CSV 文件，提取所有唯一的 wikipedia_title
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# 假设 'wikipedia_title' 列包含所需的标题
train_titles = set(train_df['wikipedia_title'])
test_titles = set(test_df['wikipedia_title'])

# 合并所有标题
all_titles = train_titles.union(test_titles)

total_titles = len(all_titles)
print(f"从 train.csv 和 test.csv 中共提取到 {total_titles} 个唯一的 Wikipedia 标题")

# 2. 使用 ijson 流式解析 kb_wiki 文件，筛选出标题在 all_titles 集合中的条目
filtered_entries = {}
found_count = 0

with open(kb_wiki_path, "r", encoding="utf-8") as f:
    # 假设 kb_wiki 文件的顶层结构是一个字典，键为标题，值为条目内容
    for title, entry in ijson.kvitems(f, ''):
        if title in all_titles:
            # 将匹配的条目添加到结果字典中
            filtered_entries[title] = entry
            found_count += 1
            # 如果已经找到所有标题对应的条目，可以提前结束解析
            if found_count == total_titles:
                break

# 计算检索成功率
success_rate = (found_count / total_titles) * 100 if total_titles > 0 else 0

print(f"共筛选出 {found_count} 个匹配的条目")
print(f"检索成功率: {success_rate:.2f}%")

# 3. 将筛选后的条目保存到输出文件中
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(filtered_entries, f_out, ensure_ascii=False, indent=4)

print(f"解析后的数据已保存到：{output_path}")
