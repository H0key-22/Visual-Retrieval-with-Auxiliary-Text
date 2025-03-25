import json
import torch
import faiss
import pickle
import requests
import gc
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
from concurrent.futures import ThreadPoolExecutor  # 多线程加速下载

# 1. 读取 JSON 文件
with open("image_urls.json", "r", encoding="utf-8") as file:
    image_data = json.load(file)

# 计算所有图片的总数（仅限前 100 个 URL）
url_list = list(image_data.keys())[:10]  # 限制处理前 100 个 URL
total_images = sum(len(image_data[url]) for url in url_list)

# 2. 加载 CLIP ViT-L/14 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 3. 创建 FAISS 内积索引
d = 768  # CLIP ViT-L/14 输出向量的维度
index = faiss.IndexFlatIP(d)  # 使用内积索引（适用于归一化的 CLIP 特征）

image_indices = []  # 存储每张图片所属 URL 的索引

# 多线程下载图片
def download_image(img_url):
    try:
        response = requests.get(img_url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert("RGB")
    except Exception as e:
        print(f"✗ Error downloading {img_url}: {e}")
    return None

# 全局进度条
with tqdm(total=total_images, desc="Total Progress", unit="img") as pbar:
    # 4. 遍历前 100 个 URL 并处理每个图像
    for url_idx, url in enumerate(tqdm(url_list, desc="Processing URLs", position=0)):
        image_urls = image_data[url]

        # 4.1 使用多线程下载所有图片
        with ThreadPoolExecutor(max_workers=10) as executor:
            images = list(executor.map(download_image, image_urls))  # 并行下载

        # 4.2 处理下载成功的图片
        for img in images:
            if img is None:
                continue  # 跳过下载失败的图片

            try:
                # 4.3 预处理图像
                inputs = processor(images=img, return_tensors="pt").to(device)

                # 4.4 计算 CLIP 图像特征并归一化
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # 归一化向量（IndexFlatIP 需要）
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy().astype("float32")

                # 4.5 添加到 FAISS 索引
                index.add(image_features)
                image_indices.append(url_idx-1)  # 记录该图片属于哪个 URL

                # 更新全局进度条
                pbar.update(1)

            except Exception as e:
                print(f"✗ Error processing image: {e}")

        # 释放内存
        torch.cuda.empty_cache()
        gc.collect()

# 5. 保存 FAISS 索引和 URL 索引文件
faiss.write_index(index, "clip_image_index.faiss")

with open("image_indices.pkl", "wb") as f:
    pickle.dump(image_indices, f)

with open("url_list.json", "w", encoding="utf-8") as f:
    json.dump(url_list, f, indent=4, ensure_ascii=False)

print("\n✅ Indexing completed! Faiss index, image indices, and URL list saved.")
