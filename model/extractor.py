import faiss
import numpy as np
import spacy
import torch
from sentence_transformers import SentenceTransformer
import re
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor, CLIPTokenizer


class FeatureExtractor:
    """
    根据描述返回目标物体的主要外部特征
    (Return the main external features of the target object based on its description)
    """

    def __init__(self, model_name="sbert"):
        """
        初始化特征提取器
        Args:
            model_name: 使用的模型名称，可选 "clip", "eva-clip", "sbert" (默认 "sbert")
        """
        self.model_name = model_name.lower()
        if self.model_name == "clip":
            # 加载 CLIP 模型
            model_path = "/root/autodl-tmp/CLIP"
            self.model = CLIPModel.from_pretrained(model_path)
            self.processor = CLIPProcessor.from_pretrained(model_path)
            self.model.to("cuda").eval()
        elif self.model_name == "eva-clip":
            model_path = "/path/to/model_name"
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
            self.model.to("cuda").eval()
            self.processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
            self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        elif self.model_name == "sbert":
            self.model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
            self.processor = None  # SentenceTransformer 内部处理文本
        else:
            raise ValueError(f"Unknown model_name: {model_name}")

    def split_text_into_blocks(self, text, block_size=200, step=100):
        """
        将文本按照句子进行切分，再组合成不超过 block_size 字符数的文本块，并按照 step 步长滑动窗口
        Splits text into sentences and then combines them into blocks with maximum block_size characters,
        using a sliding window with step size (in characters).

        Args:
            text: 输入文本，可以为字符串或字符串列表
            block_size: 每个文本块的最大字符数
            step: 滑动窗口步长（字符数）
        Returns:
            blocks: 文本块列表
        """
        blocks = []
        # 如果 text 是字符串，则转换为列表，统一处理
        if isinstance(text, str):
            text = [text]

        for entry in text:
            # 使用正则表达式按中文句号、感叹号、问号进行分割，同时保留标点符号
            # 按句子分割（保留标点）
            sentences = re.split(r'(?<=[.!?])', entry)
            sentences = [s.strip() for s in sentences if s.strip()]

            # 计算平均每句的字符数，进而计算步进多少句
            avg_sentence_length = len(entry) / len(sentences) if sentences else 1
            step_sentences = max(1, int(step / avg_sentence_length))

            sentence_start = 0  # 滑动窗口起始位置

            while sentence_start < len(sentences):
                current_block = []
                current_length = 0
                i = sentence_start

                # 组合句子，直到达到 block_size 限制
                while i < len(sentences) and current_length + len(sentences[i]) <= block_size:
                    current_block.append(sentences[i])
                    current_length += len(sentences[i])
                    i += 1

                if current_block:
                    blocks.append("".join(current_block))
                else:
                    # 如果连单句都不能加入，直接将该句加入，避免死循环
                    blocks.append(sentences[sentence_start])
                    i = sentence_start + 1

                # 滑动窗口移动
                sentence_start += step_sentences

        return blocks

    def encode_text(self, texts):
        """
        将文本编码为向量
        Args:
            texts: 文本列表
        Returns:
            numpy 数组形式的文本向量
        """
        if self.model_name == "clip":
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            # 将输入移动到模型所在设备
            device = next(self.model.parameters()).device
            inputs = {key: val.to(device) for key, val in inputs.items()}
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
            embeddings = embeddings.cpu().numpy()
        elif self.model_name == "eva-clip":
            if isinstance(texts, list):
                # 使用 tokenizer 对文本进行批量 tokenization
                input_ids = self.tokenizer(texts, return_tensors="pt", padding=True).input_ids.to('cuda')
                # 直接对批量文本进行编码
                with torch.no_grad(), torch.cuda.amp.autocast():
                    embeddings = self.model.encode_text(input_ids)
            else:
                # 单个文本处理
                input_ids = self.tokenizer([texts], return_tensors="pt", padding=True).input_ids.to('cuda')
                with torch.no_grad(), torch.cuda.amp.autocast():
                    embeddings = self.model.encode_text(input_ids)
            # 转换为 numpy 数组
            embeddings = embeddings.cpu().numpy()

        elif self.model_name == "sbert":
            # SentenceTransformer 默认返回 numpy 数组，如果 convert_to_tensor=False
            embeddings = self.model.encode(texts, convert_to_tensor=False)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        # 计算 L2 范数
        norms = np.linalg.norm(embeddings, axis=-1, keepdims=True)
        # 避免除零，进行归一化
        embeddings /= (norms + 1e-10)  # 加上一个小数避免除以零

        return embeddings

    def retrieve_blocks(self, query_question, text, k):
        """
        根据查询问题，使用 FAISS 检索方法从文本中找出最相关的 k 个文本块
        Args:
            query_question: 查询问题字符串
            text: 输入文本（字符串或字符串列表）
            k: 需要检索的文本块数量
        Returns:
            best_blocks: 与查询最相关的文本块列表
        """
        # 将文本切分为多个块
        blocks = self.split_text_into_blocks(text)
        if not blocks:
            return []
        # 对每个文本块编码
        encoded_blocks = self.encode_text(blocks)
        d = encoded_blocks.shape[1]
        # 创建 FAISS 索引（基于 L2 距离）
        index = faiss.IndexFlatL2(d)
        # FAISS 要求数据为连续内存且类型为 float32
        index.add(np.ascontiguousarray(encoded_blocks.astype('float32')))

        # 对查询问题进行编码
        query_vector = self.encode_text(query_question)
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = np.ascontiguousarray(query_vector.astype('float32'))

        # 执行检索
        distances, indices = index.search(query_vector, k)
        best_blocks = [blocks[i] for i in indices[0] if i < len(blocks)]
        return best_blocks

    def extract_feature(self, text, object, k=10):
        """
        提取目标物体描述中的主要外部特征，返回与查询最相关的 k 个文本块
        Args:
            text: 目标物体的描述文本（字符串或字符串列表）
            k: 返回的文本块数量
        Returns:
            features: 主要外部特征的文本块列表
        """
        query_text = "What are the prominent external characteristics of {}?".format(object)
        features = self.retrieve_blocks(query_text, text, k)
        return features

    def extract_from_query(self, text, query_text, k=10):
        """
        根据自定义查询从文本中提取相关信息
        Args:
            text: 输入文本（字符串或字符串列表）
            query_text: 自定义查询问题
            k: 返回的文本块数量
        Returns:
            features: 与查询最相关的文本块列表
        """
        features = self.retrieve_blocks(query_text, text, k)
        return features

    def text_similarity(self, text_1, text_2_list):
        # 对单个文本和文本列表分别编码
        embed_1 = self.encode_text(text_1)  # shape: (d,)
        embed_2 = self.encode_text(text_2_list)  # shape: (n, d), n为列表中文本数量

        # 对单个文本进行L2归一化
        norm_embed_1 = embed_1 / np.linalg.norm(embed_1, ord=2)
        if text_2_list == []:
            return 0
        # 对文本列表的每个文本进行L2归一化（按行归一化）
        if embed_2.ndim == 1:
            norm_embed_2 = embed_2 / np.linalg.norm(embed_2)
        else:
            norm_embed_2 = embed_2 / np.linalg.norm(embed_2, axis=1, keepdims=True)

        # 计算余弦相似度，结果是一个数组，每个元素对应一个文本的相似度
        similarity = np.dot(norm_embed_2, norm_embed_1)
        # 返回最大的相似度
        return np.max(similarity)


def extract_target_theme(question):
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
    doc = nlp(question)

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


# 示例用法
if __name__ == "__main__":
    description = (
        "This is a detailed textual description of the external characteristics of a target object. "
        "It may include various attributes such as color, shape, size, texture, and other distinguishing features. "
        "For instance, apples have a bright red hue, a smooth and glossy surface, and a perfectly round shape. "
        "Its dimensions could be moderate, neither too large nor too small, making it easy to hold in one’s hand. "
        "Additionally, it might exhibit subtle patterns, such as fine lines or embossed details, adding to its uniqueness. "
        "Other notable aspects could include its weight, reflective properties, or any distinctive markings that set it apart. "
        "In some cases, the description might also mention how the object interacts with light, whether it is transparent, "
        "translucent, or opaque. Furthermore, the description may provide comparisons to familiar objects, "
        "helping to convey a clearer picture of its overall appearance."
    )
    # Initialize the feature extractor, options are "sbert", "clip", or "eva-clip"
    extractor = FeatureExtractor(model_name="sbert")
    prominent_features = extractor.extract_feature(description, "apple", k=2)

    print("Main external features:")
    for feature in prominent_features:
        print(feature)

