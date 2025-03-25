from sentence_transformers import SentenceTransformer, util

# 预训练的SBERT模型
model = SentenceTransformer('all-MiniLM-L6-v2')
print("本地模型加载成功！")

# 给定文本
query_text = "如何提高自然语言处理的效果？"

# 需要比较的文本列表
candidate_texts = [
    "怎样提升NLP模型的性能？",
    "今天天气很好，我们去公园吧。",
    "深度学习可以帮助改善文本分类的准确率。",
    "请问自然语言处理和机器翻译有什么关系？"
]

# 计算句子嵌入
query_embedding = model.encode(query_text, convert_to_tensor=True)
candidate_embeddings = model.encode(candidate_texts, convert_to_tensor=True)

# 计算相似度
similarities = util.pytorch_cos_sim(query_embedding, candidate_embeddings)[0]

# 找出最相似的文本
most_similar_index = similarities.argmax().item()
most_similar_text = candidate_texts[most_similar_index]

print(f"最相似的文本是: {most_similar_text}")
