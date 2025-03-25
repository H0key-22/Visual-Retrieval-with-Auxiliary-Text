import json
import re
from argparse import ArgumentParser
import faiss
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel, AutoModel, AutoProcessor
import tqdm
from model import WikipediaKnowledgeBaseEntry
from utils import load_csv_data, get_test_question
from model import Agent


def answer_retrieve(
    test_file_path: str,
    knowledge_base_path: str,
    model: str,
    query_num: int,
    top_k: int
):

    if model == "clip":
        # 加载 CLIP 模型
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    else:
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

    # 加载Agent
    answer_agent = Agent()

    def evaluation(prediction_list, ground_truth_list):
        """
        检查 ground_truth_list 中的每个元素是否出现在 prediction_list 里。
        如果出现返回 1，否则返回 0。

        参数:
            prediction_list (list): 预测结果列表
            ground_truth_list (list): 真实值列表

        返回:
            0或者1
        """
        for truth in ground_truth_list:
            for pred in prediction_list:
                if truth in pred:  # 只要 truth 是 pred 的子串，就返回 1
                    return 1
        return 0

    def split_text_into_blocks(evidence_bace, block_size=200, step=100):
        blocks = []
        for key in evidence_bace.keys():
            text = evidence_bace[key]
            for i in range(0, len(text) - block_size + 1, step):
                blocks.append(text[i:i + block_size])
        return blocks

    # 使用 CLIP 对文本块进行编码
    def encode_text(texts):
        inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model.get_text_features(**inputs)
        return embeddings.numpy()

    def combine_titles_and_texts(section_titles, section_texts):
        # 检查长度是否一致，不一致则抛出 ValueError
        if len(section_titles) != len(section_texts):
            raise ValueError("Error: section_titles and section_texts must have the same length!")

        # 组合成字典
        return {title: text for title, text in zip(section_titles, section_texts)}

    def retrieve_blocks(query_question, query_frame, evidence_base, k):
        blocks = split_text_into_blocks(evidence_base)
        encoded_blocks = np.array(encode_text(blocks))

        # 创建 FAISS 索引
        d, = encoded_blocks.shape[1:]
        index = faiss.IndexFlatL2(d)
        index.add(encoded_blocks)

        best_blocks_dict = {}  # 用于存储唯一文本及其最高相似度

        for query in query_frame:
            query_vector = encode_text([query]).reshape(1, -1)

            # 进行 FAISS 检索，返回前 k 个文本块
            D, I = index.search(query_vector, k)

            # 遍历搜索结果
            for similarity, idx in zip(D[0], I[0]):
                text = blocks[idx]  # 获取文本块

                # 只保留相似度更高的相同文本
                if text not in best_blocks_dict or similarity > best_blocks_dict[text]:
                    best_blocks_dict[text] = similarity  # 更新最高相似度

        # 按相似度降序排序，并保留前 k 个
        sorted_blocks = sorted(best_blocks_dict.items(), key=lambda x: x[1], reverse=True)[:k]

        # 提取最终的文本部分
        best_blocks_frame = [text for text, _ in sorted_blocks]

        # 处理question查询
        query_vector = encode_text([query_question]).reshape(1, -1)

        # 进行 FAISS 检索，返回前 k 个最相关的文本块
        D, I = index.search(query_vector, k)

        # 获取最相关的前 k 个文本块
        best_blocks_question = [blocks[i] for i in I[0]]

        return best_blocks_question, best_blocks_frame

    def match(text):
        matches = re.findall(r"\(\w+\)\s*(.*)", text)
        return matches

    test_list, test_header = load_csv_data(test_file_path)
    test_list = test_list[:10]
    kb_dict = json.load(open(knowledge_base_path, "r"))

    recall_question = 0
    recall_frame = 0

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

        # 构建答案模板
        response = answer_agent.generate_answers(query_question, query_num, target_obj)
        query_frame = match(response)

        best_blocks_question, best_blocks_frame = retrieve_blocks(
            query_question, query_frame, evidence_base, top_k)

        ground_truth = example["answer"]
        ground_truth_list = ground_truth.split("|")

        recall_question += evaluation(best_blocks_question, ground_truth_list)
        recall_frame += evaluation(best_blocks_frame, ground_truth_list)

        # 输出最相关的前 5 个文本块
        print("与最相关的前 5 个文本块:")
        for i, block in enumerate(best_blocks_question[:5], 1):
            print(f"{i}. {block}\n")

        print("与框架最相关的前 5 个文本块:")
        for i, block in enumerate(best_blocks_frame[:5], 1):
            print(f"{i}. {block}\n")

        print("Question Avg Recall@{}: ", recall_question / (it + 1))
        print("Question Avg Recall@{}: ", recall_frame / (it + 1))

    return None


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--query_num", type=int, required=True)
    parser.add_argument("--top_k", type=int, required=True)

    args = parser.parse_args()

    config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "model_name": args.model,
        "query_num": args.query_num,
        "top_k": args.top_k
    }
    print("test_config: ", config)
    answer_retrieve(**config)