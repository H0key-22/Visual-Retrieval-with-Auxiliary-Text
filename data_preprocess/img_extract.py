import json
import torch
import faiss
import pickle
import requests
import numpy as np
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel

# 1. 读取 JSON 文件，获取每个 URL 的 image_urls
with open("image_urls.json", "r", encoding="utf-8") as file:
    image_data = json.load(file)

# 2. 加载 CLIP ViT-L/14 模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# 3. 创建 FAISS 内积索引
d = 768  # CLIP ViT-L/14 输出向量的维度
index = faiss.IndexFlatIP(d)  # 使用内积索引（适用于归一化的 CLIP 特征）

url_list = list(image_data.keys())  # URL 列表
image_indices = []  # 存储每张图片所属 URL 的索引

# 4. 遍历 URL 并处理每个图像
for url_idx, url in enumerate(tqdm(url_list, desc="Processing URLs")):
    image_urls = image_data[url]

    for img_url in tqdm(image_urls, desc=f"Processing {url_idx}", leave=False):
        try:
            # 4.1 下载图像
            response = requests.get(img_url, timeout=10)
            if response.status_code == 200:
                image = Image.open(BytesIO(response.content)).convert("RGB")

                # 4.2 预处理图像
                inputs = processor(images=image, return_tensors="pt").to(device)

                # 4.3 计算 CLIP 图像特征并归一化
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # 归一化向量（IndexFlatIP 需要）
                image_features /= image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.cpu().numpy().astype("float32")

                # 4.4 添加到 FAISS 索引
                index.add(image_features)
                image_indices.append(url_idx)  # 记录该图片属于哪个 URL

                print(f"✓ Encoded and indexed: {img_url} (Belongs to URL Index: {url_idx})")

        except Exception as e:
            print(f"✗ Error processing {img_url}: {e}")

# 5. 保存 FAISS 索引和 URL 索引文件
faiss.write_index(index, "clip_image_index.faiss")

with open("image_indices.pkl", "wb") as f:
    pickle.dump(image_indices, f)

with open("url_list.json", "w", encoding="utf-8") as f:
    json.dump(url_list, f, indent=4, ensure_ascii=False)

print("\n✅ Indexing completed! Faiss index, image indices, and URL list saved.")
