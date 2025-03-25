import os
import json
import shutil
import pandas as pd

# 文件路径设置
csv_file = '../inat_split.csv'
json_file = '../val_id2name.json'
dest_dir = 'E:/INat/inspect'
Inat_path = "E:/INat"  # 基础路径

# 如果目标文件夹不存在，则创建
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# 读取CSV文件，假设文件中有一列名为 'dataset_image_ids'
df = pd.read_csv(csv_file)
image_ids = df['dataset_image_ids'].tolist()
image_ids = image_ids[4:10]

# 加载JSON文件，获得id到图片路径的映射
with open(json_file, 'r') as f:
    id2name = json.load(f)

# 遍历每个 image id
for img_id in image_ids:
    img_id_str = str(img_id)  # JSON中的key通常是字符串
    if img_id_str in id2name:
        # 拼接成完整的图片路径：基础路径 + JSON中的相对路径
        src_path = os.path.join(Inat_path, id2name[img_id_str])
        # 检查图片文件是否存在
        if os.path.exists(src_path):
            # 复制图片到目标文件夹，文件名保持不变
            shutil.copy(src_path, os.path.join(dest_dir, os.path.basename(src_path)))
            print(f"已复制 {src_path} 到 {dest_dir}")
        else:
            print(f"图片文件不存在：{src_path}")
    else:
        print(f"ID {img_id_str} 未在JSON映射中找到。")
