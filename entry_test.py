import ijson
import json

# 文件路径设置
kb_wiki_path = r"E:\encyclopedic_kb_wiki.json"

# 定义一个字典，用于存储标题为 "Cornus suecica" 的条目
filtered_entries = {}
found_count = 0

# 使用 ijson 流式解析 kb_wiki 文件，逐个检测条目的 title 字段
with open(kb_wiki_path, "r", encoding="utf-8") as f:
    # 假设文件的顶层结构为字典，遍历所有键值对
    for wiki_url, entry in ijson.kvitems(f, ''):
        if entry.get("title") == "Cornus suecica":
            # 提取需要的字段，若字段不存在则使用默认值
            filtered_entry = {
                "title": entry.get("title"),
                "section_titles": entry.get("section_titles", []),
                "section_texts": entry.get("section_texts", []),
                "image_urls": entry.get("image_urls", []),
                "image_reference_descriptions": entry.get("image_reference_descriptions", []),
                "image_section_indices": entry.get("image_section_indices", []),
                "url": entry.get("url")
            }
            filtered_entries[wiki_url] = filtered_entry
            found_count += 1

# 打印筛选后的结果
print(f"共找到 {found_count} 个标题为 'Cornus suecica' 的条目")
print(json.dumps(filtered_entries, ensure_ascii=False, indent=4))
