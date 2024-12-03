# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification, DataCollatorForSeq2Seq
from sklearn.preprocessing import LabelEncoder
from peft import LoraConfig, get_peft_model, TaskType
from transformers import TrainingArguments, Trainer
import json
from seqeval.metrics import f1_score, classification_report
import numpy as np

label_list = ["I-其他治疗", "B-西医诊断", "B-中医治则", "I-中医治则", "I-西医诊断", "I-中医诊断", "B-中医证候", "B-西医治疗",
              "I-中药", "B-其他治疗", "B-方剂", "I-中医证候", "I-临床表现", "I-中医治疗", "B-中医治疗", "I-西医治疗", "B-中医诊断",
              "B-中药", "B-临床表现", "I-方剂", "O"]
ner_dict = {'I-其他治疗':0,
            'B-西医诊断':1,
            'B-中医治则':2,
            'I-中医治则':3,
            'I-西医诊断':4,
            'I-中医诊断':5,
            'B-中医证候':6,
            'B-西医治疗':7,
            'I-中药':8,
            'B-其他治疗':9,
            'B-方剂':10,
            'I-中医证候':11,
            'I-临床表现':12,
            'I-中医治疗':13,
            'B-中医治疗':14,
            'I-西医治疗':15,
            'B-中医诊断':16,
            'B-中药':17,
            'B-临床表现':18,
            'I-方剂':19,
            'O':20}

print(ner_dict)

def process_data(input_file, output_file):
    sentences, labels = [], []
    sentence, label = [], []

    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            print(line)
            if not line:  # 句子结束
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                sentence, label = [], []
            else:
                line_size = len(line.split())
                if line_size == 1:
                    sentence.append(" ")
                    label.append(20)
                    continue
                word, tag = line.split()
                sentence.append(word)
                label.append(ner_dict[tag])

    if sentence:  # 防止最后一个句子未处理
        sentences.append(sentence)
        labels.append(label)

    # 转换为SFT格式
    sft_data = [{"tokens": s, "ner_tags": l} for s, l in zip(sentences, labels)]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=4)


# 示例：处理训练集和验证集
# process_data("medical.train", "medical_train.json")
# process_data("medical.dev", "medical_dev.json")
# process_data("medical.test", "medical_test.json")
print("finish handle data.")

def validate_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

        for entry in data:
            if 'tokens' not in entry or 'ner_tags' not in entry:
                raise ValueError("Missing 'tokens' or 'ner_tags' field in JSON entry.")
            if len(entry['tokens']) != len(entry['ner_tags']):
                raise ValueError(f"Length mismatch between 'text' and 'labels' in entry: {entry}")


# 验证训练集和验证集
validate_json('medical_train.json')
validate_json('medical_dev.json')
validate_json('medical_test.json')

print("JSON 文件格式正确")

# 加载基座大模型
# model_name = "hfl/chinese-macbert-base"
model_name = "Qwen/Qwen1.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.padding_side = "right"  # 一定要设置padding_side为right，否则batch大于1时可能不收敛
dataset = load_dataset("json", data_files={"train": "medical_train.json", "validation": "medical_dev.json"})

label_encoder = LabelEncoder()
label_encoder.fit(list(range(21)))

def preprocess_function(examples):
    # 对文本进行分词
    tokenized_examples = tokenizer(examples["tokens"], truncation=True, max_length=200, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_examples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_examples["labels"] = labels

    return tokenized_examples


tokenized_dataset = dataset.map(preprocess_function, batched=True)

for i in range(1):
    print(tokenized_dataset["train"][i])
    print(len(tokenized_dataset["train"][i]['tokens']))
    print(len(tokenized_dataset["train"][i]['ner_tags']))
    print(len(tokenized_dataset["train"][i]['input_ids']))
    print(len(tokenized_dataset["train"][i]['attention_mask']))
    # print(len(tokenized_dataset["train"][i]['token_type_ids']))


# 配置LoRA
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS,
    r=8,  # LoRA rank
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.1,  # dropout
    bias="none",  # 不调整 bias
)

model = AutoModelForTokenClassification.from_pretrained(model_name,
                                                        num_labels=len(label_list),
                                                        low_cpu_mem_usage=True,
                                                        torch_dtype=torch.bfloat16,
                                                        # device_map="auto",
                                                        load_in_4bit=True,
                                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                                        bnb_4bit_quant_type="nf4",
                                                        bnb_4bit_use_double_quant=True)

# 应用LoRA到模型
model = get_peft_model(model, lora_config)

model.enable_input_require_grads()

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    logging_steps=10, # log 打印的频率
    eval_strategy="epoch", # 评估策略
    learning_rate=2e-5, # 学习率
    per_device_train_batch_size=8,
    num_train_epochs=4,
    save_strategy="epoch", # 保存策略
    load_best_model_at_end=True,
    metric_for_best_model="f1", # 设定评估指标
    gradient_accumulation_steps=32,
    gradient_checkpointing=True,
    optim="paged_adamw_32bit"
)


# from seqeval.metrics import f1_score, classification_report
seqeval = evaluate.load("seqeval_metric.py")

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)

    # 将id转换为原始的字符串类型的标签
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")

    return {
        "f1": result["overall_f1"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)

# 开始训练
# 在训练期间捕获和输出调试信息
trainer.train()

# 计算F1 Score
results = trainer.evaluate()
print(results)

# 将LoRA权重合并到主模型
model = model.merge_and_unload()

# 保存合并后的模型
model.save_pretrained("./merged_model")

# 如果需要保存对应的tokenizer
tokenizer.save_pretrained("./merged_model")
