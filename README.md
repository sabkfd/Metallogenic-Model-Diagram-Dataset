# Metallogenic Model Diagram Dataset & VLM Fine-Tuning

> **[中文版](README.zh.md)** | English

This project focuses on multi-task fine-tuning of Vision-Language Models (VLMs) in the geological mineralization domain. It includes training code, structured datasets, and metallogenic model diagram image resources. Training is conducted using LoRA fine-tuning based on Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct base models.

---

## Directory Structure

```
Data/
├── Code/                                    # Training code
│   ├── Qwen2.5-VL-7B – Training.py          # Qwen2.5-VL-7B fine-tuning script
│   └── Qwen3-VL-8B – Training.py            # Qwen3-VL-8B fine-tuning script
├── data/                                    # Structured datasets
│   ├── dataset.json                         # Multi-task annotated dataset in JSON
│   └── chengkuang_dataset.xlsx              # Dataset in Excel format
├── Metallogenic Model Diagram – Training/   # Training images (156 images)
│   ├── 001.jpg
│   └── ...                                  # Supports jpg / jpeg / png formats
├── Metallogenic Model Diagram – Test/       # Test images (21 images)
│   ├── test_001.png
│   └── ...                                  # Supports png / jpeg formats
└── README.md
```

---

## Directory Descriptions

### 1. `Code/` — Training Code

Contains LoRA fine-tuning scripts for Qwen2.5-VL-7B and Qwen3-VL-8B vision-language models. Both scripts share the same structure, differing only in base model loading and some configuration details.

**Core Modules:**

- **Metric Computation (`compute_metrics`)**: Uses `jieba` tokenization + `rouge_chinese` for ROUGE-1/ROUGE-L, and `nltk` for BLEU-4, used for generative evaluation.
- **Sample Processing (`process_func`)**: Dispatches to different task branches based on `task_type`, constructing unified VLM message inputs and labels. Constraints: `MAX_PIXELS=1284512`, `MAX_LENGTH=5120`.
- **Dataset Expansion (`expand_dataset`)**: Dynamically expands a single JSON sample into multiple training samples (QA, mask filling, long/short captions, etc.), enabling one-image multi-task training.
- **Model Loading & LoRA Configuration**: Downloads base models via `modelscope`, injects LoRA adapters via `peft.LoraConfig`.

**Key Training Hyperparameters:**

| Parameter | Value |
| :--- | :--- |
| Base Model | Qwen2.5-VL-7B-Instruct / Qwen3-VL-8B-Instruct |
| Fine-tuning Strategy | LoRA |
| LoRA `r` / `alpha` / `dropout` | 64 / 64 / 0.1 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Training Epochs | 10 |
| Learning Rate | 1e-5 |
| Per-Device Batch Size | 1 |
| Gradient Accumulation Steps | 16 |
| Precision | bf16 |
| Eval/Save Strategy | Per epoch |
| Best Model Metric | `rouge-l` |
| Max Generation Length | 2560 |
| Validation Set Size | 20 |

**Task Dispatch Logic (routed by `task_type`):**

| Task Type | task_type | Trigger Condition |
| :--- | :--- | :--- |
| Short QA | `short_qa` | `qa_pairs` with `len(answer) <= 100` |
| Deep Reasoning QA | `reasoning_qa` | `qa_pairs` with `len(answer) > 100` |
| Pre-generated Mask Fill | `mask_fill_pre` | `mask_paragraphs` exists |
| Dynamic Mask Fill | `mask_fill_dynamic` | Both `caption1` and `keywords` exist |
| Image Captioning | `caption` | `caption1` or `caption2` exists |

**Experiment Tracking:** Uses SwanLab to log training progress (project names: `Qwen2.5-VL-finetune` / `Qwen3-VL-finetune`).

---

### 2. `data/` — Structured Dataset

Contains structured data for multi-task fine-tuning in both JSON and Excel formats:
- `dataset.json`: The primary data file loaded by training scripts; each record corresponds to one image and its multi-task annotations.
- `chengkuang_dataset.xlsx`: Excel spreadsheet version of the same data, for manual review and editing.

## Dataset Schema Documentation

### Overview
This dataset is specifically designed for the multi-task fine-tuning of Vision-Language Models (VLMs) in the geological and mineralization domain. Each JSON object represents a single image sample along with its associated multi-modal annotations, including comprehensive descriptions, keyword extractions, question-answering (QA) pairs, and fill-in-the-blank (masking) tasks.

### Field Descriptions

#### Core Fields (Required for all samples)
- **`image_path`** *(string)*
  - **Description**: The file path (absolute or relative) to the source image (e.g., a geological mineralization pattern diagram).
  - **Usage**: Loaded as the visual input for the VLM.

- **`caption1`** *(string)*
  - **Description**: The primary, comprehensive, and detailed textual description of the image. It typically contains in-depth geological analysis and professional terminology.
  - **Usage**: Used as the target output for the "Long Captioning" task and serves as the base text for the "Dynamic Mask Filling" task.

- **`keywords`** *(list of strings)*
  - **Description**: A list of core geological terms, entities, and concepts extracted from the image and its descriptions.
  - **Usage**: Used to dynamically generate masking tasks by replacing these keywords with `[Mask]` tokens.

#### Optional Fields (Available for specific subsets)
- **`caption2`** *(string, optional)*
  - **Description**: A secondary, more concise summary or short description of the image.
  - **Usage**: Used as the target output for the "Short Captioning" task.

- **`qa_pairs`** *(list of objects, optional)*
  - **Description**: A list of Question-Answer pairs designed to test the model's understanding and reasoning capabilities regarding the image.
  - **Sub-fields**:
    - **`question`** *(string)*: The specific question asked about the image content.
    - **`answer`** *(string)*: The ground-truth answer.
  - **Usage**: Mapped to QA tasks. The length of the `answer` string automatically determines the task routing during training:
    - *Short QA*: Answer length $\le$ 100 characters (direct, concise answers).
    - *Deep Reasoning QA*: Answer length > 100 characters (requires systematic analysis and long-form reasoning).

- **`mask_paragraphs`** *(list of objects, optional)*
  - **Description**: Pre-generated text paragraphs where specific geological terms have been replaced with `[Mask]` tokens for fill-in-the-blank tasks.
  - **Sub-fields**:
    - **`masked_caption`** *(string)*: The paragraph text containing `[Mask]` placeholders.
    - **`masked_words`** *(list of strings)*: The exact original words that were replaced by the masks (used for evaluation/reference).
    - **`candidates`** *(list of strings)*: A restricted list of candidate words provided to the model to choose from.
    - **`all_keywords`** *(list of strings)*: A broader list of all relevant keywords associated with this specific paragraph context.
  - **Usage**: Mapped to the "Pre-generated Mask Filling" task, where the model must select the correct terms from the candidates to fill the masks.

---

### Task Mapping
The dataset fields are dynamically expanded into specific training tasks during the data loading phase:

| Task Type | Trigger Condition | Input (Prompt) | Target (Label) |
| :--- | :--- | :--- | :--- |
| **Short QA** | `qa_pairs` exists & `len(answer) <= 100` | Image + Short Question | Short Answer |
| **Reasoning QA** | `qa_pairs` exists & `len(answer) > 100` | Image + Reasoning Question | Long Reasoning Answer |
| **Mask Fill (Pre)** | `mask_paragraphs` exists | Image + Masked Text + Candidates | Original Unmasked Text |
| **Mask Fill (Dynamic)**| `caption1` & `keywords` exist | Image + Caption1 with random `[Mask]` | Original Caption1 |
| **Long Captioning** | `caption1` exists | Image + "Describe the image" | Caption1 |
| **Short Captioning** | `caption2` exists | Image + "Describe the image" | Caption2 |

---

### Example JSON Object

```json
{
  "image_path": "/path/to/geology_diagram_001.jpg",
  "caption1": "该成矿模式图展示了典型的热液脉型矿床形成过程。富含金属离子的热液流体沿断裂带向上运移，在物理化学条件突变处发生沉淀成矿...",
  "caption2": "热液脉型矿床成矿模式图。",
  "keywords": ["热液脉型", "断裂带", "沉淀成矿", "流体运移"],
  "qa_pairs": [
    {
      "question": "图中热液流体主要沿什么构造运移？",
      "answer": "断裂带。"
    },
    {
      "question": "请详细分析该图中的成矿物理化学条件变化及其对矿物沉淀的影响。",
      "answer": "根据图示，热液流体在深部处于高温高压环境，随着流体沿断裂带向上运移，温度和压力逐渐降低。当流体到达浅部裂隙发育区时，由于温压骤降以及可能与地下水混合导致pH值变化，流体中的金属离子溶解度急剧下降，从而发生大规模的沉淀成矿作用..."
    }
  ],
  "mask_paragraphs": [
    {
      "masked_caption": "该成矿模式图展示了典型的[Mask]矿床形成过程。富含金属离子的热液流体沿[Mask]向上运移...",
      "masked_words": ["热液脉型", "断裂带"],
      "candidates": ["热液脉型", "斑岩型", "断裂带", "背斜轴部"],
      "all_keywords": ["热液脉型", "断裂带", "沉淀成矿"]
    }
  ]
}
```

---

### 3. `Metallogenic Model Diagram – Training/` — Training Images

Contains metallogenic model diagrams used for model training, totaling **156 images**, named from `001` to `156`, in `.jpg` / `.jpeg` / `.png` formats. These images are the visual input sources referenced by the `image_path` field in `dataset.json`.

### 4. `Metallogenic Model Diagram – Test/` — Test Images

Contains test metallogenic model diagrams for model inference and evaluation, totaling **21 images**, named `test_001` through `test_021`, in `.png` / `.jpeg` formats. This directory is independent of the training set and is used to evaluate the generalization capability of the fine-tuned model.

---

## Dependencies

The training scripts require the following main libraries:

```
torch
transformers
peft
datasets
modelscope
qwen_vl_utils
swanlab
Pillow
jieba
rouge_chinese
nltk
numpy
```

---

## Quick Start

1. Install the dependencies listed above.
2. Modify `dataset_file_path` in the training script to point to `data/dataset.json` in this project.
3. Modify `output_dir` to a local model output directory.
4. Run the training script for the desired base model:

```bash
python "Code/Qwen2.5-VL-7B – Training.py"
# or
python "Code/Qwen3-VL-8B – Training.py"
```

5. Monitor ROUGE / BLEU metrics in real time via the SwanLab dashboard during training.