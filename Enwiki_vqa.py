import csv
import ijson
import json

# 文件路径设置
kb_wiki_path = r"E:\encyclopedic_kb_wiki.json"
input_files = [
    r"E:\train.csv",
    r"E:\test.csv"
]
# 新的过滤后 CSV 文件路径（分别对应原始文件）
filtered_files = [
    r"E:\train_filter.csv",
    r"E:\test_filter.csv"
]
output_path = r"E:\encyclopedic_kb_wiki_vqa.json"

# 1. 提取所有 wikipedia_title（使用 csv 模块读取 CSV 文件），同时过滤掉 question_type 为 "2_hop" 的行
title_set = set()

# 如果后续需要对 CSV 文件进行再次过滤，也可以先将所有行读取到内存中（可选）
csv_rows = {}

for file_path in input_files:
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        csv_rows[file_path] = rows
        for row in rows:
            # 如果 question_type 为 "2_hop" 则跳过该行
            if row.get("question_type") == "2_hop":
                continue
            title = row.get("wikipedia_title")
            if title:
                title_set.add(title)

print(f"从 {len(input_files)} 个 CSV 文件中共提取到 {len(title_set)} 个 wikipedia_title（已过滤 question_type 为 2_hop 的行）")

# 2. 使用 ijson 流式解析 kb_wiki 文件，根据条目的 title 进行过滤
parsed_subset = {}
found_count = 0

with open(kb_wiki_path, "r", encoding="utf-8") as f:
    # 假设 kb_wiki 文件的顶层结构是一个字典，键为 URL，值为条目内容
    for wiki_url, entry in ijson.kvitems(f, ''):
        title = entry.get("title")
        if title in title_set:
            parsed_entry = {
                "title": title,
                "section_titles": entry.get("section_titles", []),
                "section_texts": entry.get("section_texts", []),
                "image_urls": entry.get("image_urls", []),
                "image_reference_descriptions": entry.get("image_reference_descriptions", []),
                "image_section_indices": entry.get("image_section_indices", []),
                "url": entry.get("url")
            }
            # 使用 title 作为 key 保存
            parsed_subset[title] = parsed_entry
            found_count += 1
            # 如果已经找到所有目标条目，可以提前退出
            if found_count == len(title_set):
                break

print(f"共筛选出 {found_count} 个条目")

# 3. 输出成功筛选的比率
if len(title_set) > 0:
    ratio = found_count / len(title_set)
    print(f"成功筛选比率：{ratio:.2%}")
else:
    print("没有提取到任何 wikipedia_title")

# 4. 计算并输出未筛选到的条目列表
not_found_titles = list(title_set - set(parsed_subset.keys()))
print("未筛选到的条目：", not_found_titles)

# 5. 将筛选后的条目保存到输出文件中
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(parsed_subset, f_out, ensure_ascii=False, indent=4)
print(f"解析后的数据已保存到：{output_path}")

# 6. 针对每个 CSV 文件，筛选出 wikipedia_title 在 parsed_subset 中的行，
#    并将结果写入对应的 *_filter.csv 文件中
for input_file, filtered_file in zip(input_files, filtered_files):
    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(filtered_file, "w", encoding="utf-8", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            # 同样跳过 question_type 为 "2_hop" 的行
            if row.get("question_type") == "2_hop":
                continue
            if row.get("wikipedia_title") in parsed_subset:
                writer.writerow(row)
    print(f"已保存筛选后的 CSV 文件：{filtered_file}")
