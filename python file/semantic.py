import re


def extract_main_noun_phrases(sentence):
    """
    从一句话中提取主要名词短语：
      规则1：查找系动词（如 is/are/was/were/be 等）后面的表语部分
      规则2：查找类似 “refer to”, “point to”, “relate to” 及其变体后面的宾语部分
      清理名词短语，应用以下规则：
      规则3：删除从句部分 —— 移除从 relative pronoun（that, which, who, whom, whose, where, when）开始后的内容
      规则4：删除后置定语 —— 移除从常见介词（of, with, in, at, by, for）开始后的内容
    任意规则匹配成功，都直接返回该规则后面的部分，
    并对提取的部分应用规则3和规则4进行清理（删除从句和后置定语）。
    """

    def clean_noun_phrase(np):
        """
        清理名词短语，应用以下规则：
          规则3：删除从句部分 —— 移除从 relative pronoun（that, which, who, whom, whose, where, when）开始后的内容
          规则4：删除后置定语 —— 移除从常见介词（of, with, in, at, by, for）开始后的内容
        """
        # 规则3：删除从句部分
        np = re.sub(r'\b(?:that|which|who|whom|whose|where|when)\b.*', '', np, flags=re.IGNORECASE).strip()
        # 规则4：删除后置定语部分
        np = re.sub(r'\b(?:of|with|in|at|by|for)\b.*', '', np, flags=re.IGNORECASE).strip()
        return np
    noun_phrases = []
    # 按逗号、分号、句号拆分为子句
    clauses = re.split(r'[;,.]', sentence)


    # 系动词模式（大小写不敏感）
    linking_verbs = r'\b(?:is|are|was|were|be|been|being|seem|seems|seemed|became|become|becomes)\b'
    linking_regex = re.compile(linking_verbs, re.IGNORECASE)

    # 动词短语模式，考虑常见时态和形式（例如: refers, referred, referring）
    vp_regex = re.compile(r'\b(?:refer(?:s|red|ring)?|point(?:s|ed|ing)?|relate(?:s|d|ing)?)\s+to\b', re.IGNORECASE)

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # 规则1：查找系动词后面的内容（表语）
        m_link = linking_regex.search(clause)
        if m_link:
            complement = clause[m_link.end():].strip()
            # 清理提取到的名词短语，删除从句和后置定语部分
            complement_clean = clean_noun_phrase(complement)
            noun_phrases.append(complement_clean)

        # 规则2：查找指定动词短语后面的内容（宾语）
        m_vp = vp_regex.search(clause)
        if m_vp:
            obj_complement = clause[m_vp.end():].strip()
            # 同样清理提取到的宾语部分
            obj_complement_clean = clean_noun_phrase(obj_complement)
            noun_phrases.append(obj_complement_clean)

    return noun_phrases


# ----------------- 测试样例 -----------------

test_sentences = [
    # 样例1：仅使用规则1，含有从句，需要删除
    (
        "The main issue is the problem that refers to the outdated policy.",
        ["the problem"]
    ),
    # 样例2：仅使用规则2，含有后置定语，需要删除
    (
        "Another challenge points to the underlying cause with a complex background.",
        ["the underlying cause"]
    ),
    # 样例3：使用规则1，含有从句和后置定语同时存在
    (
        "Her concern is the decision which was influenced by many factors in the recent years.",
        ["the decision"]
    ),
    # 样例4：使用规则1，先删除从句，再删除后置定语
    (
        "The report is the analysis of the situation that reveals a hidden trend.",
        ["the analysis"]
    ),
    # 样例5：使用规则2，含有后置定语
    (
        "This challenge points to the risk with significant implications.",
        ["the risk"]
    ),
    # 样例6：使用规则1，含有后置定语（无从句）
    (
        "Our objective is the plan which outlines a strategy of growth.",
        ["the plan"]
    ),
    # 样例7：一个句子中含有两个提取部分：规则1和规则2
    (
        "A significant problem was the error in the system, which pointed to a misconfiguration.",
        ["the error", "a misconfiguration"]
    ),
    # 样例8：复杂句子，通过逗号分隔，仅提取第一个有效的表语部分
    (
        "The key factor is the decision, which was made under pressure, that led to the downfall.",
        ["the decision"]
    ),
    # 样例9：含有 to‐infinitive结构，规则1提取后，再清理后置定语
    (
        "His suggestion was to revise the policy for better compliance with the new regulations.",
        ["to revise the policy"]
    ),
]

# 运行测试样例
for idx, (sentence, expected) in enumerate(test_sentences, start=1):
    result = extract_main_noun_phrases(sentence)
    print(f"样例 {idx}:")
    print("输入句子：", sentence)
    print("提取结果：", result)
    print("预期结果：", expected)
    print("-" * 50)
