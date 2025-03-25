import ijson
import json

# 文件路径设置
kb_wiki_path = r"E:\encyclopedic_kb_wiki.json"
retrieval_result_files = [
    r"E:\yanzuWu\retrieval_result_glv2.json",
    r"E:\yanzuWu\retrieval_result_inat.json"
]
output_path = r"E:\encyclopedic_kb_wiki_parsed.json"

# 1. 加载 retrieval_result 文件，提取所有 URL
url_set = set()

for file_path in retrieval_result_files:
    with open(file_path, "r", encoding="utf-8") as f:
        retrieval_result = json.load(f)
    for data in retrieval_result.values():
        urls = data.get("retrieved_entries", [])
        url_set.update(urls)

print(f"从 {len(retrieval_result_files)} 个 retrieval_result 文件中共提取到 {len(url_set)} 个 URL")

# 2. 使用 ijson 流式解析 kb_wiki 文件，筛选出 URL 在 url_set 中的条目
parsed_subset = {}
found_count = 0

with open(kb_wiki_path, "r", encoding="utf-8") as f:
    # 遍历顶层的键值对（假设 kb_wiki 文件的顶层结构是一个字典）
    for wiki_url, entry in ijson.kvitems(f, ''):
        if wiki_url in url_set:
            # 按照需求提取各字段，若字段缺失则使用默认值
            parsed_entry = {
                "title": entry.get("title"),
                "section_titles": entry.get("section_titles", []),
                "section_texts": entry.get("section_texts", []),
                "image_urls": entry.get("image_urls", []),
                "image_reference_descriptions": entry.get("image_reference_descriptions", []),
                "image_section_indices": entry.get("image_section_indices", []),
                "url": entry.get("url")
            }
            parsed_subset[wiki_url] = parsed_entry
            found_count += 1
            # 如果已经找齐了所有 URL，可以提前结束解析
            if found_count == len(url_set):
                break

print(f"共筛选出 {found_count} 个条目")

# 3. 将筛选后的条目保存到输出文件中
with open(output_path, "w", encoding="utf-8") as f_out:
    json.dump(parsed_subset, f_out, ensure_ascii=False, indent=4)

print(f"解析后的数据已保存到：{output_path}")
