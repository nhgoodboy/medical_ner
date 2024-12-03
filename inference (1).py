import json
import torch
from transformers import Trainer, AutoModelForTokenClassification, AutoTokenizer
from datasets import Dataset

# 加载合并后的模型
model = AutoModelForTokenClassification.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# 验证模型是否可用
model.eval()
print("模型加载成功！")

# 从 JSON 文件加载测试数据
with open('medical_test.json', 'r', encoding='utf-8') as f:
    test_data = json.load(f)

# 提取测试文本
test_texts = [item['tokens'] for item in test_data]

# 对测试集进行编码
test_encodings = tokenizer(test_texts, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True)

# 创建测试集数据集
test_dataset = Dataset.from_dict({"input_ids": test_encodings["input_ids"], "attention_mask": test_encodings["attention_mask"]})

# 使用 Trainer 进行推理
trainer = Trainer(
    model=model,
    tokenizer=tokenizer
)

# 执行推理
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)

label_list = ["I-其他治疗", "B-西医诊断", "B-中医治则", "I-中医治则", "I-西医诊断", "I-中医诊断", "B-中医证候", "B-西医治疗",
              "I-中药", "B-其他治疗", "B-方剂", "I-中医证候", "I-临床表现", "I-中医治疗", "B-中医治疗", "I-西医治疗", "B-中医诊断",
              "B-中药", "B-临床表现", "I-方剂", "O"]

# 转换预测结果为标签
model.config.id2label = {idx: label for idx, label in enumerate(label_list)}
id2label = model.config.id2label
predicted_labels = []

# 根据 attention_mask 去掉填充部分
for pred_seq, mask in zip(preds, test_encodings["attention_mask"]):
    pred_labels = [id2label[pred] for pred, m in zip(pred_seq, mask) if m == 1]
    predicted_labels.append(pred_labels)

for text, pred in zip(test_texts, predicted_labels):
    print(f"文本: {text}")
    print(f"预测标签: {pred}")
    print(f"预测标签长度: {len(pred)}")
