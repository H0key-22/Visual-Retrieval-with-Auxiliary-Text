import faiss

# 读取 FAISS 索引文件
index = faiss.read_index("clip_image_index.faiss")

# 查看索引信息
print(f"索引的向量维度: {index.d}")
print(f"索引中的向量数量: {index.ntotal}")

# 如果索引是可训练的，可以查看训练状态
if index.is_trained:
    print("索引已训练")
else:
    print("索引未训练")


# 获取索引的前 5 个向量
for i in range(min(5, index.ntotal)):
    vector = index.reconstruct(i)
    print(f"向量 {i}: {vector}")

# 读取 FAISS 索引文件
index = faiss.read_index("E:\evqa_2M_faiss_index\evqa_index_full\kb_index.faiss")

# 查看索引信息
print(f"索引的向量维度: {index.d}")
print(f"索引中的向量数量: {index.ntotal}")

# 如果索引是可训练的，可以查看训练状态
if index.is_trained:
    print("索引已训练")
else:
    print("索引未训练")


# 获取索引的前 5 个向量
for i in range(min(5, index.ntotal)):
    vector = index.reconstruct(i)
    print(f"向量 {i}: {vector}")