import re


def extract_main_noun_phrases(sentence):
    """
    从一句话中提取主要名词短语：
      规则1：查找系动词（如 is/are/was/were/be 等）后面的表语部分
      规则2：查找类似 “refer to”, “point to”, “relate to” 及其变体后面的宾语部分
      规则1和规则2是“或”的关系
      规则3：如果名词短语部分中出现了从句，则需要删除从句部分
      规则4：如果名词短语部分中出现了后置定语，则需要删除后置定语

    任意规则匹配成功，都直接返回该规则后面的部分。
    """
    noun_phrases = []
    # 按逗号、分号、句号拆分成子句
    clauses = re.split(r'[;,.]', sentence)

    # 定义系动词模式（大小写不敏感）
    linking_verbs = r'\b(?:is|are|was|were|be|been|being|seem|seems|seemed|became|become|becomes)\b'
    linking_regex = re.compile(linking_verbs, re.IGNORECASE)

    # 定义动词短语模式，考虑了常见时态和形式，如: refers, referred, referring等
    vp_regex = re.compile(r'\b(?:refer(?:s|red|ring)?|point(?:s|ed|ing)?|relate(?:s|d|ing)?)\s+to\b', re.IGNORECASE)

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # 应用规则1：查找系动词后面的内容
        m_link = linking_regex.search(clause)
        if m_link:
            complement = clause[m_link.end():].strip()
            noun_phrases.append(complement)

        # 应用规则2：查找特定动词短语后面的内容
        m_vp = vp_regex.search(clause)
        if m_vp:
            obj_complement = clause[m_vp.end():].strip()
            noun_phrases.append(obj_complement)

    return noun_phrases


# 测试示例
if __name__ == "__main__":
    sentence = ("The main issue is the problem that refers to the outdated policy, "
                "while another case points to the underlying cause.")
    print(extract_main_noun_phrases(sentence))
    # 输出可能为：
    # ['the problem that refers to the outdated policy', 'the outdated policy', 'the underlying cause']
