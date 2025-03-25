import json
import re
from argparse import ArgumentParser
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
import tqdm
from model import WikipediaKnowledgeBaseEntry, FeatureExtractor
from utils import load_csv_data, get_test_question


def answer_retrieve(
    test_file_path: str,
    knowledge_base_path: str,
    model_name: str,
    top_k: int
):

    if model_name == "clip":
        # 加载 CLIP 模型
        model = CLIPModel.from_pretrained("/root/autodl-tmp/CLIP")
        processor = CLIPProcessor.from_pretrained("/root/autodl-tmp/CLIP")
        model.to("cuda").eval()
    elif model_name == "eva-clip":
        model_path = "/path/to/model_name"
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        del model.text_projection
        del model.text_model  # avoiding OOM
        model.to("cuda").eval()
        processor = AutoProcessor.from_pretrained(
            "openai/clip-vit-large-patch14")
    elif model_name == "sbert":
        model = SentenceTransformer('all-MiniLM-L6-v2')
    

    def evaluation(prediction_list, ground_truth_list):
        """
        检查 ground_truth_list 中的每个元素是否出现在 prediction_list 里。
        如果出现返回 1，否则返回 0。
        如果 truth 为 "A&&B" 的形式，则需要分别判断 A 和 B 是否都在 pred 中出现

        参数:
            prediction_list (list): 预测结果列表
            ground_truth_list (list): 真实值列表

        返回:
            0 或 1
        """
        for truth in ground_truth_list:
            for pred in prediction_list:
                # 如果 truth 包含 "&&"，则拆分后判断每个部分是否都出现在 pred 中
                if '&&' in truth:
                    parts = truth.split('&&')
                    # 去除每个部分两边可能的空白字符
                    parts = [part.strip() for part in parts]
                    if all(part in pred for part in parts):
                        return 1
                # 如果 truth 不包含 "&&"，直接判断 truth 是否在 pred 中
                else:
                    if truth in pred:
                        return 1
        return 0

    def split_text_into_blocks(evidence_base, block_size=200, step=100):
        blocks = []

        for key in evidence_base:
            text = key

            # 1️⃣ 按句子分割（保留标点）
            sentences = re.split(r'(?<=[。！？])', text)
            sentences = [s.strip() for s in sentences if s.strip()]  # 去除空白

            # 2️⃣ 组合句子，形成不超过 block_size 的文本块
            current_block = []
            current_length = 0
            sentence_start = 0  # 滑动窗口起始位置

            while sentence_start < len(sentences):
                for i in range(sentence_start, len(sentences)):
                    sentence = sentences[i]
                    if current_length + len(sentence) > block_size:
                        break
                    current_block.append(sentence)
                    current_length += len(sentence)

                # 添加当前 block
                if current_block:
                    blocks.append("".join(current_block))

                # 3️⃣ 滑动窗口，按照 step 移动
                sentence_start += step // (len(text) / len(sentences))  # 近似等效 step 控制
                current_block = []
                current_length = 0

        return blocks

    # 使用 CLIP 对文本块进行编码
    def encode_text(texts):
        if model_name == "clip":
            inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                embeddings = model.get_text_features(**inputs)

        elif model_name == "eva-clip":
            inputs = processor(texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
            # 迁移到设备（CPU/GPU）
            inputs = {key: val.to("cpu") for key, val in inputs.items()}
            # 获取文本嵌入
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state[:, 0, :]

        elif model_name == "sbert":
            embeddings = model.encode(texts, convert_to_tensor=True)

        return embeddings.numpy()


    def combine_titles_and_texts(section_titles, section_texts):
        # 检查长度是否一致，不一致则抛出 ValueError
        if len(section_titles) != len(section_texts):
            raise ValueError("Error: section_titles and section_texts must have the same length!")

        # 返回列表，每个元素格式为 "title: text"
        return [f"{title}: {text}" for title, text in zip(section_titles, section_texts)]


    def retrieve_blocks(query_question,evidence_base, k):
        blocks = split_text_into_blocks(evidence_base)
        encoded_blocks = np.array(encode_text(blocks))

        # 创建 FAISS 索引
        d, = encoded_blocks.shape[1:]
        index = faiss.IndexFlatL2(d)
        index.add(encoded_blocks)

        # 处理question查询
        query_vector = encode_text([query_question]).reshape(1, -1)

        # 进行 FAISS 检索，返回前 k 个最相关的文本块
        D, I = index.search(query_vector, k)

        # 获取最相关的前 k 个文本块
        best_blocks_question = [blocks[i] for i in I[0]]

        return best_blocks_question


    test_list, test_header = load_csv_data(test_file_path)
    kb_dict = json.load(open(knowledge_base_path, "r"))

    recall_question = 0
    recall_gt = 0
    recall_question_5 = 0
    recall_question_10 = 0
    recall_question_20 = 0

    for it, test_example in tqdm.tqdm(enumerate(test_list)):
        example = get_test_question(it, test_list, test_header)
        target_url = example["wikipedia_url"]
        entry = WikipediaKnowledgeBaseEntry(kb_dict[target_url])
        section_titles = entry.section_titles
        section_texts = entry.section_texts
        evidence_base = combine_titles_and_texts(section_titles, section_texts)

        # 查询问题
        query_question = example["question"]
        target_obj = example["wikipedia_title"]

        best_blocks_question = retrieve_blocks(
            query_question, evidence_base, top_k)

        ground_truth = example["answer"]
        ground_truth_list = ground_truth.split("|")

        recall_question_5 += evaluation(best_blocks_question[:5], ground_truth_list)
        recall_question_10 += evaluation(best_blocks_question[:10], ground_truth_list)
        recall_question_20 += evaluation(best_blocks_question[:20], ground_truth_list)
        recall_gt += evaluation(section_texts, ground_truth_list)

        print("\n")
        print("Question: {}".format(query_question))
        print("Ground Truth: {}".format(ground_truth))

        # 输出最相关的前 5 个文本块
        print("与最相关的前 5 个文本块:")
        for i, block in enumerate(best_blocks_question[:5], 1):
            print(f"{i}. {block}\n")

        print("Retrieve 5 Avg Recall@{}: {}".format(it, recall_question_5 / (it + 1)))
        print("Retrieve 10 Avg Recall@{}: {}".format(it, recall_question_10 / (it + 1)))
        print("Retrieve 20 Avg Recall@{}: {}".format(it, recall_question_20 / (it + 1)))
        print("Ground_truth Avg Recall@{}: {}".format(it, recall_gt / (it + 1)))

    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--top_k", type=int, required=True)

    args = parser.parse_args()

    config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "model_name": args.model_name,
        "top_k": args.top_k
    }
    print("test_config: ", config)
    answer_retrieve(**config)