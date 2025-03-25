import pickle
from collections import Counter

with open("image_indices.pkl", "rb") as f:
    data = pickle.load(f)

print(f"数据类型: {type(data)}")

# 确保数据是列表
if isinstance(data, list):
    # 统计每个整数的出现次数
    counter = Counter(data[:1000])

    # 按数值排序输出
    sorted_counts = sorted(counter.items())  # 按数值排序

    print("整数出现次数统计：")
    for value, count in sorted_counts:
        print(f"数值 {value}: 出现 {count} 次")
else:
    print("数据不是列表类型，无法统计整数频率")