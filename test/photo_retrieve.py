import PIL
from data_utils import targetpad_transform
from argparse import ArgumentParser
import json, tqdm
from model import (
    ClipRetriever,
    WikipediaKnowledgeBaseEntry,
)
from utils import load_csv_data, get_test_question, get_image, remove_list_duplicates

iNat_image_path = "/PATH/TO/INAT_ID2NAME"


def eval_recall(candidates, ground_truth, top_ks=[1, 5, 10, 20, 100]):
    recall = {k: 0 for k in top_ks}
    for k in top_ks:
        if ground_truth in candidates[:k]:
            recall[k] = 1
    return recall


def run_test(
    test_file_path: str,
    knowledge_base_path: str,
    faiss_index_path: str,
    top_ks: list,
    retrieval_top_k: int,
    **kwargs
):
    test_list, test_header = load_csv_data(test_file_path)
    with open(iNat_image_path + "/val_id2name.json", "r") as f:
            iNat_id2name = json.load(f)
        
    retriever = ClipRetriever(device="cuda:0", model=kwargs["retriever_vit"])
    # retriever.save_knowledge_base_faiss(knowledge_base_path, scores_path=score_dict, save_path=faiss_index_path)
    retriever.load_knowledge_base(knowledge_base_path)
    retriever.load_faiss_index(faiss_index_path)

    recalls = {k: 0 for k in top_ks}

    retrieval_result = {}
    for it, test_example in tqdm.tqdm(enumerate(test_list)):
        example = get_test_question(it, test_list, test_header)
        image = PIL.Image.open(
            get_image(
                example["dataset_image_ids"].split("|")[0],
                example["dataset_name"],
                iNat_id2name
            )
        )
        ground_truth = example["wikipedia_url"]
        if example["dataset_name"] == "infoseek":
            data_id = example["data_id"]
        else:
            data_id = "E-VQA_{}".format(it)
        print("wiki_url: ", example["wikipedia_url"])
        print("question: ", example["question"])

        top_k = retriever.retrieve_image_faiss(image, top_k=retrieval_top_k)
        top_similarity = top_k[0]["similarity"]
        print(top_similarity)
        top_k_wiki = [retrieved_entry["url"] for retrieved_entry in top_k]
        top_k_wiki = remove_list_duplicates(top_k_wiki)
        entries = [retrieved_entry["kb_entry"] for retrieved_entry in top_k]
        entries = remove_list_duplicates(entries)
        seen = set()
        retrieval_simlarities = [
            top_k[i]["similarity"]
            for i in range(retrieval_top_k)
            if not (top_k[i]["url"] in seen or seen.add(top_k[i]["url"]))
        ]
        if kwargs["save_result"]:
            retrieval_result[data_id] = {
                "retrieved_entries": [entry.url for entry in entries[:20]],
                "retrieval_similarities": [
                    sim.item() for sim in retrieval_simlarities[:20]
                ],
            }
        else:
            recall = eval_recall(top_k_wiki, ground_truth, top_ks)
            for k in top_ks:
                recalls[k] += recall[k]
        for k in top_ks:
            print("Avg Recall@{}: ".format(k), recalls[k] / (it + 1))



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--knowledge_base", type=str, required=True)
    parser.add_argument("--faiss_index", type=str, required=True)
    parser.add_argument(
        "--top_ks",
        type=str,
        default="1,5,10,20,100",
        help="comma separated list of top k values, e.g. 1,5,10,20,100",
    )
    parser.add_argument("--retrieval_top_k", type=int, default=20)
    parser.add_argument("--save_result", action="store_true")
    parser.add_argument("--save_result_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--retriever_vit", type=str, default="clip", help="clip or eva-clip"
    )
    args = parser.parse_args()

    test_config = {
        "test_file_path": args.test_file,
        "knowledge_base_path": args.knowledge_base,
        "faiss_index_path": args.faiss_index,
        "top_ks": [int(k) for k in args.top_ks.split(",")],
        "retrieval_top_k": args.retrieval_top_k,
        "save_result": args.save_result,
        "save_result_path": args.save_result_path,
        "resume_from": args.resume_from,
        "retriever_vit": args.retriever_vit,
    }
    print("test_config: ", test_config)
    run_test(**test_config)
