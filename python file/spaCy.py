import spacy


def extract_target_theme(sentence):
    """
    从问句中提取目标主题（target theme），例如：
      - "How big is an adult of this animal?"  返回 "animal"
      - "In addition to woodland, where else is the brown-throated parakeet found?" 返回 "parakeet"

    使用以下启发式规则：
      1. 如果存在 "this"/"these"，则提取其后紧跟的名词；
      2. 如果句子中存在带有 "of" 介词短语，则提取其宾语作为目标主题；
      3. 查找句子中的主语（nsubj 或 nsubjpass）的名词短语；
      4. 回退：返回句子中最后出现的名词短语的核心词。
    """
    # 加载英文模型
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)

    # 规则1：查找 "this" 或 "these"，并提取紧跟其后的名词
    for token in doc:
        if token.text.lower() in ("this", "these"):
            # 尝试获取下一个 token
            if token.i + 1 < len(doc):
                next_token = doc[token.i + 1]
                if next_token.pos_ in ("NOUN", "PROPN"):
                    return next_token.text

    # 规则2：检查名词短语中是否存在 "of" 介词短语，提取其宾语
    for chunk in doc.noun_chunks:
        for token in chunk:
            if token.dep_ == "prep" and token.text.lower() == "of":
                # 寻找其子节点中依赖关系为宾语 (pobj)
                for child in token.children:
                    if child.dep_ == "pobj":
                        return child.text

    # 规则3：查找句子中的主语（nsubj 或 nsubjpass）
    subject_chunks = [chunk for chunk in doc.noun_chunks
                      if any(token.dep_ in ("nsubj", "nsubjpass") for token in chunk)]
    if subject_chunks:
        # 如果有多个主语，优先返回含有形容词修饰的名词短语
        for chunk in subject_chunks:
            if any(tok.pos_ == "ADJ" for tok in chunk):
                return chunk.root.text
        return subject_chunks[0].root.text

    # 规则4：回退：返回句子中最后出现的名词短语的核心词
    noun_chunks = list(doc.noun_chunks)
    if noun_chunks:
        return noun_chunks[-1].root.text

    return None


def extract_main_noun_phrases(sentence):
    """
    从一句话中提取所有主要名词短语，并去除后置修饰。
    规则：
      1. 对于文档中的每个句子，取句子根节点的所有子节点中依存关系为 "attr" 的 token，
         并以该 token 的左右边界提取完整的名词短语。
      2. 遍历该名词短语中的 token，若遇到以下情况，则截断后续部分：
            - 遇到介词短语(token.dep_ == "prep")且其子树中含有实体类型 DATE、TIME、GPE 或 LOC。
            - 遇到标点（如逗号），后面紧跟着相对代词（"who", "that", "which"）。
            - 遇到相对从句(token.dep_ == "relcl")，作为备选规则（如果上述标点规则未命中）。
      3. 返回所有主要名词短语组成的列表。
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    results = []

    for sent in doc.sents:
        root = sent.root
        # 遍历根节点的子节点，寻找依存关系为 "attr" 的 token
        for child in root.children:
            if child.dep_ == "attr":
                # 提取以该 token 为核心的完整子树作为初步名词短语
                main_span = doc[child.left_edge.i: child.right_edge.i + 1]
                tokens = list(main_span)
                cut_index = len(tokens)  # 默认保留全部

                for i, token in enumerate(tokens):
                    # 规则1：检测逗号后紧跟相对代词（who/that/which）的情况
                    if token.text == "," and i + 1 < len(tokens) and tokens[i + 1].text.lower() in {"who", "that",
                                                                                                    "which"}:
                        cut_index = i
                        break
                    # 规则2：检测介词短语中包含 DATE/TIME/GPE/LOC 实体
                    if token.dep_ == "prep":
                        if any(child.ent_type_ in {"DATE", "TIME", "GPE", "LOC"} for child in token.subtree):
                            cut_index = i
                            break
                    # 规则3：检测相对从句（relcl），作为备选截断点
                    if token.dep_ == "relcl":
                        cut_index = i
                        break

                trimmed_span = doc[tokens[0].i: tokens[0].i + cut_index]
                results.append(trimmed_span.text)

    return results

if __name__ == '__main__':
    examples = [
        "How big is an adult of this animal?",
        "In addition to woodland, where else is the brown-throated parakeet found?",
        "Is this cat common in urban areas?",
        "Are these birds migratory?"
    ]

    for sentence in examples:
        target = extract_target_theme(sentence)
        print(f"句子：{sentence}\n提取的目标主题：{target}\n")

    examples = [
        "The $100,000 infield was the infield of the Philadelphia Athletics in the early 1910s.",
        "!!! (/tʃ(ɪ)k.tʃ(ɪ)k.tʃ(ɪ)k/ ch(i)k-ch(i)k-ch(i)k), also known as Chk Chk Chk, is an American rock band from Sacramento, California, formed in 1996 by lead singer Nic Offer.",
        # 可添加更多示例：
        "The quick brown fox is the fox of the local forest in summer.",
        "!PAUS3, or THEE PAUSE, (born July 27, 1981) is an international platinum selling musician and artist, who began his career in his early teens in the former Soviet Bloc nations of Ukraine, Romania and Bulgaria.",
        "Jennifer F. Provencher (born 22 October 1979) is a Canadian conservation biologist."
    ]

    for sentence in examples:
        main_np = extract_main_noun_phrases(sentence)
        print(f"句子：{sentence}\n主要名词短语：{main_np}\n")
