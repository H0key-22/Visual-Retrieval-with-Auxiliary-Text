from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# 加载模型和处理器
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = AutoModelForImageTextToText.from_pretrained("Salesforce/blip-image-captioning-large")

# 读取图片
img_path = "E:/INat/inspect/4898d9b4-0881-48c3-97ba-e81816feab68.jpg"
raw_image = Image.open(img_path).convert('RGB')

# 条件图像描述生成
text = "Detailed feature of the plant:"
inputs = processor(raw_image, text, return_tensors="pt")
out = model.generate(**inputs)
print("条件生成结果：", processor.decode(out[0], skip_special_tokens=True))

# 无条件图像描述生成
inputs = processor(raw_image, return_tensors="pt")
out = model.generate(**inputs)
print("无条件生成结果：", processor.decode(out[0], skip_special_tokens=True))
