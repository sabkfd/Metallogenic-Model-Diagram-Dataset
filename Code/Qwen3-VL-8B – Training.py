import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from swanlab.integration.transformers import SwanLabCallback
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
)
from PIL import Image
import os
import swanlab
import json
import random
import numpy as np
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from torch.utils.data import Subset


# --- 1. Metric computation function ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    try:
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    except Exception as e:
        print(f"⚠️ Decoding error: {e}")
        return {"rouge-l": 0.0, "bleu-4": 0.0}

    decoded_preds_cut = [" ".join(jieba.cut(pred)) for pred in decoded_preds]
    decoded_labels_cut = [" ".join(jieba.cut(label)) for label in decoded_labels]

    result = {}
    rouge = Rouge()
    try:
        if len(decoded_preds_cut) > 0:
            rouge_scores = rouge.get_scores(decoded_preds_cut, decoded_labels_cut, avg=True)
            result["rouge-1"] = rouge_scores["rouge-1"]["f"]
            result["rouge-l"] = rouge_scores["rouge-l"]["f"]
    except Exception as e:
        print(f"ROUGE computation error: {e}")
        result["rouge-l"] = 0.0

    bleu_scores = []
    smooth = SmoothingFunction().method1
    for pred, label in zip(decoded_preds_cut, decoded_labels_cut):
        pred_tokens = pred.split()
        label_tokens = [label.split()]
        if not pred_tokens:
            bleu_scores.append(0.0)
            continue
        score = sentence_bleu(label_tokens, pred_tokens, smoothing_function=smooth)
        bleu_scores.append(score)

    result["bleu-4"] = np.mean(bleu_scores) if bleu_scores else 0.0
    return {k: round(v, 4) for k, v in result.items()}


# --- 2. Core processing function (standard processing logic, with the bbox branch removed) ---
def process_func(example):
    MAX_PIXELS = 1284512
    MAX_LENGTH = 5120

    # Image paths are unified, so use them directly
    file_path = example["image_path"]

    try:
        image_obj = Image.open(file_path).convert("RGB")
    except Exception as e:
        print(f"Failed to read image: {file_path}, error: {e}")
        return None

    task_type = example.get("task_type", "caption")

    # ==================== Task dispatch logic ====================

    # ---- Task 1: Concise QA (short answers, distinguished by answer length <= 100) ----
    if task_type == "short_qa":
        question = example.get("specific_question", "请描述图片内容")
        user_question = f"【地质问答】{question}"
        final_caption = example.get("specific_answer", "")
        system_text = "你是一个地质学专家，请根据图片内容直接给出答案。"

    # ---- Task 2: Deep reasoning QA (long reasoning, distinguished by answer length > 100) ----
    elif task_type == "reasoning_qa":
        question = example.get("specific_question", "请分析图片中的地质过程")
        user_question = f"【地质问答】{question}"
        final_caption = example.get("specific_answer", "")
        system_text = "你是一个资深地质成矿分析专家。请根据地质模式图，进行系统性分析推理，给出完整的结论。"

    # ---- Task 3: Pre-generated mask filling (using mask_paragraphs) ----
    elif task_type == "mask_fill_pre":
        masked_caption = example.get("masked_caption", "")
        candidates = example.get("candidates", [])
        all_keywords = example.get("all_keywords", [])

        candidate_text = ""
        if candidates:
            candidate_text = f"\n可选词汇：{', '.join(candidates)}"
        if all_keywords:
            candidate_text += f"\n所有相关关键词：{', '.join(all_keywords)}"

        user_question = f"【地质术语补全】请根据图片内容，从以下词汇中选择合适的词补全描述中的[Mask]标记：{candidate_text}\n\n{masked_caption}"
        final_caption = example.get("specific_answer", example.get("caption1", ""))
        system_text = "你是一个地质学专家，请从给定的候选词汇中选择最合适的词补全[Mask]标记。"

    # ---- Task 4: Dynamic mask filling (dynamically generated) ----
    elif task_type == "mask_fill_dynamic":
        # Prefer caption1 (long description) for dynamic masking
        masked_caption = example.get("caption1", example.get("caption", ""))
        keywords = example.get("keywords", [])
        valid_keywords = [k for k in keywords if k in masked_caption]
        if valid_keywords:
            mask_count = max(1, int(len(valid_keywords) * 0.5 + 0.5))
            keywords_to_mask = random.sample(valid_keywords, min(mask_count, len(valid_keywords)))
            for kw in keywords_to_mask:
                masked_caption = masked_caption.replace(kw, "[Mask]", 1)
        user_question = f"【地质术语补全】请根据图片内容，补全以下描述中的[Mask]标记：\n{masked_caption}"
        final_caption = example.get("caption1", example.get("caption", ""))
        system_text = "你是一个地质学专家，擅长根据地质成矿模式图补全专业术语描述。"

    # ---- Default / fallback: comprehensive caption description ----
    else:
        user_question = "请描述图片内容"
        final_caption = example.get("caption1", example.get("caption", ""))
        system_text = "你是一个专业的地质分析助手。"

    # Construct the message body
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj},
                {"type": "text", "text": str(user_question)},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": str(final_caption)}]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
        max_pixels=MAX_PIXELS,
        min_pixels=3136,
    )

    input_ids = inputs["input_ids"][0]
    attention_mask = inputs["attention_mask"][0]

    # Labels processing
    prompt_messages = messages[:2]
    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    prompt_inputs = processor(
        text=[prompt_text], images=image_inputs, videos=video_inputs,
        padding=False, return_tensors="pt",
        max_pixels=MAX_PIXELS,
        min_pixels=3136,
    )
    prompt_length = len(prompt_inputs["input_ids"][0])

    labels = input_ids.clone()
    if prompt_length < len(labels):
        labels[:prompt_length] = -100
    else:
        labels[:] = -100
    labels[input_ids == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs['pixel_values'].to(torch.bfloat16),
        "image_grid_thw": inputs['image_grid_thw'].squeeze(0)
    }


# --- 3. Dataset class ---
class QwenDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.raw_data = raw_data

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        result = process_func(self.raw_data[i])
        # Filter out samples that fail to load
        if result is None:
            return self.__getitem__((i + 1) % len(self.raw_data))
        return result


# --- 4. Dataset expansion function (adapted to the single-file structure) ---
def expand_dataset(data_list):
    expanded = []

    for item in data_list:
        image_path = item.get("image_path", "")
        caption1 = item.get("caption1", item.get("caption", ""))
        caption2 = item.get("caption2", "")
        keywords = item.get("keywords", [])

        # ====== Task 1: Process QA pairs ======
        if 'qa_pairs' in item and item['qa_pairs']:
            for pair in item['qa_pairs']:
                answer = pair.get("answer", "")
                question = pair.get("question", "")

                # [Core logic]: Distinguish QA tasks by answer length
                # Long answers correspond to reasoning QA, while short answers correspond to short QA
                if len(answer) > 100:
                    task_type = "reasoning_qa"
                else:
                    task_type = "short_qa"

                item_qa = {
                    "image_path": image_path,
                    "caption1": caption1,
                    "keywords": keywords,
                    "task_type": task_type,
                    "specific_question": question,
                    "specific_answer": answer,
                }
                expanded.append(item_qa)

        # ====== Task 2: Dynamic mask filling ======
        if caption1 and caption1.strip() and keywords:
            item_mask_dynamic = {
                "image_path": image_path,
                "caption1": caption1,
                "keywords": keywords,
                "task_type": "mask_fill_dynamic",
            }
            expanded.append(item_mask_dynamic)

        # ====== Task 3: Pre-generated mask filling ======
        if 'mask_paragraphs' in item and item['mask_paragraphs']:
            for mask_item in item['mask_paragraphs']:
                item_mask_pre = {
                    "image_path": image_path,
                    "caption1": caption1,
                    "keywords": keywords,
                    "task_type": "mask_fill_pre",
                    "masked_caption": mask_item.get("masked_caption", ""),
                    "masked_words": mask_item.get("masked_words", []),
                    "candidates": mask_item.get("candidates", []),
                    "all_keywords": mask_item.get("all_keywords", []),
                    "specific_answer": caption1,
                }
                expanded.append(item_mask_pre)

        # ====== Task 4: Comprehensive caption description ======
        # 4a. Long caption (all images have this)
        if caption1 and caption1.strip():
            item_caption1 = {
                "image_path": image_path,
                "caption1": caption1,
                "keywords": keywords,
                "task_type": "caption",
            }
            expanded.append(item_caption1)

        # 4b. Short caption2 (available for a subset of images)
        if caption2 and caption2.strip():
            item_caption2 = {
                "image_path": image_path,
                "caption1": caption2,  # Reuse the caption1 field as the target output
                "keywords": keywords,
                "task_type": "caption",
            }
            expanded.append(item_caption2)

    return expanded


# --- 5. Load dataset ---
def load_dataset_from_json(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✅ Loaded dataset: {file_path}")
    print(f"   - Total images: {len(data)}")

    # Statistics
    has_cap2 = sum(1 for item in data if item.get("caption2"))
    has_mask = sum(1 for item in data if item.get("mask_paragraphs"))
    total_qa = sum(len(item.get("qa_pairs", [])) for item in data)

    print(f"   - Images with caption2: {has_cap2}")
    print(f"   - Images with only caption1: {len(data) - has_cap2}")
    print(f"   - Images with mask_paragraphs: {has_mask}")
    print(f"   - Total QA pairs: {total_qa}")

    return data


# --- 6. Main program (adapted for Qwen3-VL-8B) ---
model_dir = snapshot_download("Qwen/Qwen3-VL-8B-Instruct", revision="master")
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_dir)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_dir, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
model.enable_input_require_grads()

# ==================== Load dataset file ====================
dataset_file_path = r"D:\ZhengHe_CKMData\archive\dataset.json"  # Path to your dataset file
raw_json_data = load_dataset_from_json(dataset_file_path)
# ====================================================================

# Split training / validation sets
random.seed(42)
random.shuffle(raw_json_data)
val_data_raw = raw_json_data[:20]
train_data_raw = raw_json_data[20:]

# Count the number of samples for each task type
expanded_train = expand_dataset(train_data_raw)
expanded_val = expand_dataset(val_data_raw)

from collections import Counter

train_tasks = Counter([e.get("task_type") for e in expanded_train])
val_tasks = Counter([e.get("task_type") for e in expanded_val])
print(f"\n📈 Training set task distribution: {dict(train_tasks)}")
print(f"📈 Validation set task distribution: {dict(val_tasks)}")

train_dataset = QwenDataset(expanded_train)
val_dataset = QwenDataset(expanded_val)
print(f"Actual training samples: {len(train_dataset)} | Actual validation samples: {len(val_dataset)}")

# LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    r=64, lora_alpha=64, lora_dropout=0.1,
)
peft_model = get_peft_model(model, config)

# Training arguments
args = Seq2SeqTrainingArguments(
    output_dir=r"D:\ZhengHe_CKMData\archive\Qwen3-VL-8B-chengkuang-3tasks",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    logging_steps=5,
    num_train_epochs=10,
    learning_rate=1e-5,
    gradient_checkpointing=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_eval_batch_size=1,
    predict_with_generate=True,
    generation_max_length=2560,
    load_best_model_at_end=True,
    metric_for_best_model="rouge-l",
    save_total_limit=3,
    bf16=True,
    remove_unused_columns=False,
)

# SwanLab
swanlab_callback = SwanLabCallback(
    project="Qwen3-VL-finetune",
    experiment_name="Qwen3-VL-8B-chengkuang",
    config={
        "dataset": "json_file",
        "source": "dataset.json",
        "metrics": "ROUGE/BLEU",
        "max_pixels": 1284512
    }
)

trainer = Seq2SeqTrainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    callbacks=[swanlab_callback],
    compute_metrics=compute_metrics,
)

# Force generation constraints
trainer.model.generation_config.max_new_tokens = 1024
trainer.model.generation_config.max_length = 5120

# Sanity check
print("\n🔍 Running Sanity Check before training...")
try:
    small_eval_ds = Subset(val_dataset, range(min(2, len(val_dataset))))
    trainer.evaluate(small_eval_ds)
    print("✅ Sanity Check passed! Evaluation logic is normal. Starting formal training...")
except Exception as e:
    print(f"❌ Sanity Check failed: {e}")
    import traceback

    traceback.print_exc()
    exit(1)

trainer.train()
trainer.save_model(os.path.join(args.output_dir, "best_model"))
swanlab.finish()