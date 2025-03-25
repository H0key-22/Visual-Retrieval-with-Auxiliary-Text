from PIL import Image
from model import ClipRetriever

def run(knowledge_base_path, faiss_index_path, **kwargs):
    retriever = ClipRetriever(device="cuda", model=kwargs["retriever_vit"])
    retriever.load_knowledge_base(knowledge_base_path)
    retriever.load_faiss_index(faiss_index_path)
    image = Image.open("img.jpg")  # 读取图片
    top_k = retriever.retrieve_image_faiss(image, top_k=5)
    print(top_k)


if __name__ == "__main__":
    app_config = {
        "retriever_vit": "clip",
        "knowledge_base_path": "E:/encyclopedic_kb_wiki_vqa.json",
        "faiss_index_path": "E:/eqva_clip_faiss_index",
    }
    run(**app_config)
