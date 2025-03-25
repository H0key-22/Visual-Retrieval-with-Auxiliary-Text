import json, tqdm
import re
from argparse import ArgumentParser
from model import extract_target_theme, extract_main_noun_phrases, WikipediaKnowledgeBaseEntry, FeatureExtractor
from utils import load_csv_data, get_test_question


def load_retrieval_results(file_path: str) -> dict:
    """
    读取retrieval_result.json文件，并返回一个字典，
    该字典以URL为键，对应的相似度（可能为列表）为值。

    参数:
        file_path (str): retrieval_result.json的文件路径。

    返回:
        dict: 形如 {url1: [sim1, sim2, ...], url2: [sim3, ...], ...}
    """
    url_to_sim = {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for data_id, result in data.items():
        entries = result.get("retrieved_entries", [])
        sims = result.get("retrieval_similarities", [])

        # 遍历当前测试样例中所有的URL和相似度对
        for url, sim in zip(entries, sims):
            if url in url_to_sim:
                url_to_sim[url].append(sim)
            else:
                url_to_sim[url] = [sim]

    return url_to_sim


def second_retrieve(
        test_file_path: str,
        knowledge_base_path: str,
        result_file_path: str,
        model_name: str,
        save_result: bool,
        save_result_path: str,
        top_ks: list
):
    def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 50]):
        recall = {k: 0 for k in top_ks}
        for k in top_ks:
            if ground_truth in candidates[:k]:
                recall[k] = 1
        return recall

    test_list, test_header = load_csv_data(test_file_path)
    test_list = test_list[:10]

    with open(result_file_path, "r", encoding="utf-8") as f:
        input_dict = json.load(f)

    extractor = FeatureExtractor(model_name)
    print("Extractor Loaded")
    kb_dict = json.load(open(knowledge_base_path, "r", encoding='utf-8'))
    print("KB Loaded")
    results = []
    recalls = {k: 0 for k in top_ks}
    recall_ref = 0

    for it, test_example in tqdm.tqdm(enumerate(test_list), total=len(test_list), desc="Test Examples"):
        example = get_test_question(it, test_list, test_header)
        target = extract_target_theme(example["question"])
        print("\n")
        print("Question:{}".format(example["question"]))
        print("Target:{}".format(target))
        ground_truth = example["wikipedia_url"]
        sim_dict = {}
        reference_urls = input_dict["E-VQA_{}".format(it)]["retrieved_entries"]

        # 为内层循环增加进度条
        for url in tqdm.tqdm(reference_urls, desc="Processing URLs", leave=False):
            entry = WikipediaKnowledgeBaseEntry(kb_dict[url])
            description = entry.section_texts[0]
            sentences = re.split(r'(?<=[.!?])\s+', description)
            first_sentence = sentences[0]
            phase_list = extract_main_noun_phrases(first_sentence)
            # 加入标题
            phase_list.append(entry.title)
            sim = extractor.text_similarity(target, phase_list)
            sim_dict[url] = sim

        # 对所有 URL 按相似度从高到低排序，并选取前 max(top_ks) 个
        max_k = max(top_ks)
        top_k_urls = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)[:max_k]

        # 将当前 example 的结果添加到 results 中
        results.append({
            'example': example,
            'top_k_urls': [
                (url, float(sim.item()) if hasattr(sim, "item") else float(sim))
                for url, sim in top_k_urls
            ]
        })

        candidate_urls = [url for url, sim in top_k_urls]
        recall = eval_recall(candidate_urls, ground_truth, top_ks)
        recall_ref_dict = eval_recall(reference_urls, ground_truth, [50])
        recall_ref += recall_ref_dict[50]
        print("\n")
        for k in top_ks:
            recalls[k] += recall[k]
            print("Avg Recall@{}: ".format(k), recalls[k] / (it + 1))
        print("Ref Recall: ", recall_ref / (it + 1))

        # 若设置了保存结果，则写入指定路径
        if save_result and save_result_path is not None:
            with open(save_result_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--result_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--save_result_path", type=str, default=None)
    args = parser.parse_args()

    config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "result_file_path": args.result_file,
        "model_name": args.model_name,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "save_result": args.save_result,
        "save_result_path": args.save_result_path,
    }
    print("config: ", config)
    second_retrieve(**config)
