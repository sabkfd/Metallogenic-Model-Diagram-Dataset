# 成矿模式图数据集与 VLM 微调训练

> **English** | [中文版](README.zh.md)

本项目面向地质成矿领域的视觉语言模型（VLM）多任务微调，包含训练代码、结构化数据集以及成矿模式图图像资源。基于 Qwen2.5-VL-7B-Instruct 与 Qwen3-VL-8B-Instruct 两个基座模型，采用 LoRA 微调策略进行训练。

---

## 📁 目录结构

```
Data/
├── Code/                                    # 训练代码
│   ├── Qwen2.5-VL-7B – Training.py          # Qwen2.5-VL-7B 微调脚本
│   └── Qwen3-VL-8B – Training.py            # Qwen3-VL-8B 微调脚本
├── data/                                    # 结构化数据集
│   ├── dataset.json                         # JSON 格式的多任务标注数据集
│   └── chengkuang_dataset.xlsx              # Excel 形式的数据集
├── Metallogenic Model Diagram – Training/   # 训练用成矿模式图（156 张）
│   ├── 001.jpg
│   └── ...                                  # 支持 jpg / jpeg / png 格式
├── Metallogenic Model Diagram – Test/       # 测试用成矿模式图（21 张）
│   ├── test_001.png
│   └── ...                                  # 支持 png / jpeg 格式
└── README.md
```

---

## 📂 各目录说明

### 1. `Code/` — 训练代码

存放 Qwen2.5-VL-7B 与 Qwen3-VL-8B 两个视觉语言模型的 LoRA 微调训练脚本。两个脚本结构一致，仅在基座模型加载与部分配置上存在差异。

**主要功能模块：**

- **指标计算（`compute_metrics`）**：使用 `jieba` 分词 + `rouge_chinese` 计算 ROUGE-1/ROUGE-L，使用 `nltk` 计算 BLEU-4，用于生成式评估。
- **样本处理（`process_func`）**：根据 `task_type` 分发到不同任务分支，统一构造 VLM 输入消息与标签。约束 `MAX_PIXELS=1284512`、`MAX_LENGTH=5120`。
- **数据集扩展（`expand_dataset`）**：将单条 JSON 样本动态展开为多条训练样本（QA、掩码填充、长短描述等），实现一图多任务训练。
- **模型加载与 LoRA 配置**：通过 `modelscope` 下载基座模型，使用 `peft.LoraConfig` 注入 LoRA 适配器。

**关键训练超参数：**

| 参数 | 取值 |
| :--- | :--- |
| 基座模型 | Qwen2.5-VL-7B-Instruct / Qwen3-VL-8B-Instruct |
| 微调策略 | LoRA |
| LoRA `r` / `alpha` / `dropout` | 64 / 64 / 0.1 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| 训练轮数 | 10 |
| 学习率 | 1e-5 |
| 单卡 batch size | 1 |
| 梯度累积步数 | 16 |
| 精度 | bf16 |
| 评估/保存策略 | 每 epoch 一次 |
| 最优模型指标 | `rouge-l` |
| 最大生成长度 | 2560 |
| 验证集样本数 | 20 |

**任务分发逻辑（按 `task_type` 路由）：**

| 任务类型 | task_type | 触发条件 |
| :--- | :--- | :--- |
| 简短问答 | `short_qa` | `qa_pairs` 中 `len(answer) <= 100` |
| 深度推理问答 | `reasoning_qa` | `qa_pairs` 中 `len(answer) > 100` |
| 预生成掩码填充 | `mask_fill_pre` | `mask_paragraphs` 存在 |
| 动态掩码填充 | `mask_fill_dynamic` | `caption1` 与 `keywords` 同时存在 |
| 图像描述 | `caption` | `caption1` 或 `caption2` 存在 |

**实验跟踪：** 使用 SwanLab 记录训练过程（项目名 `Qwen2.5-VL-finetune` / `Qwen3-VL-finetune`）。

---

### 2. `data/` — 结构化数据集

存放用于多任务微调的结构化数据，包含 JSON 与 Excel 两种形式：
- `dataset.json`：训练脚本直接加载的主数据文件，每条记录对应一张图像及其多任务标注。
- `chengkuang_dataset.xlsx`：同源数据的 Excel 表格形式，便于人工查阅与编辑。

## 📊 数据集结构文档

### 概述
本数据集专为地质成矿领域视觉语言模型（VLM）的多任务微调而设计。每个 JSON 对象代表一张图像样本及其关联的多模态标注，包括综合描述、关键词提取、问答对（QA）和填空（掩码）任务。

### 字段说明

#### 核心字段（所有样本必填）
- **`image_path`** *(字符串)*
  - **说明**：源图像（如地质成矿模式图）的文件路径（绝对或相对路径）。
  - **用途**：作为 VLM 的视觉输入加载。

- **`caption1`** *(字符串)*
  - **说明**：图像的主要、全面且详细的文本描述，通常包含深入的地质分析和专业术语。
  - **用途**：用作"长描述"任务的目标输出，并作为"动态掩码填充"任务的基础文本。

- **`keywords`** *(字符串列表)*
  - **说明**：从图像及其描述中提取的核心地质术语、实体和概念列表。
  - **用途**：通过将这些关键词替换为 `[Mask]` 标记来动态生成掩码任务。

#### 可选字段（特定子集可用）
- **`caption2`** *(字符串，可选)*
  - **说明**：图像的次要、更简洁的摘要或简短描述。
  - **用途**：用作"短描述"任务的目标输出。

- **`qa_pairs`** *(对象列表，可选)*
  - **说明**：旨在测试模型对图像内容理解和推理能力的问答对列表。
  - **子字段**：
    - **`question`** *(字符串)*：关于图像内容的具体问题。
    - **`answer`** *(字符串)*：标准答案。
  - **用途**：映射到 QA 任务。`answer` 字符串的长度在训练期间自动决定任务路由：
    - *简短问答*：答案长度 ≤ 100 字符（直接、简洁的回答）。
    - *深度推理问答*：答案长度 > 100 字符（需要系统性分析和长篇推理）。

- **`mask_paragraphs`** *(对象列表，可选)*
  - **说明**：预生成的文本段落，其中特定地质术语已被替换为 `[Mask]` 标记，用于填空任务。
  - **子字段**：
    - **`masked_caption`** *(字符串)*：包含 `[Mask]` 占位符的段落文本。
    - **`masked_words`** *(字符串列表)*：被掩码替换的原始词汇（用于评估/参考）。
    - **`candidates`** *(字符串列表)*：提供给模型选择的候选词受限列表。
    - **`all_keywords`** *(字符串列表)*：与该特定段落上下文关联的所有相关关键词的广泛列表。
  - **用途**：映射到"预生成掩码填充"任务，模型必须从候选项中选出正确的术语来填充掩码。

---

### 🔄 任务映射
数据集字段在数据加载阶段动态扩展为具体的训练任务：

| 任务类型 | 触发条件 | 输入（提示） | 目标（标签） |
| :--- | :--- | :--- | :--- |
| **简短问答** | `qa_pairs` 存在且 `len(answer) <= 100` | 图像 + 简短问题 | 简短答案 |
| **推理问答** | `qa_pairs` 存在且 `len(answer) > 100` | 图像 + 推理问题 | 长篇推理答案 |
| **掩码填充（预生成）** | `mask_paragraphs` 存在 | 图像 + 掩码文本 + 候选词 | 原始未掩码文本 |
| **掩码填充（动态）** | `caption1` 和 `keywords` 存在 | 图像 + 带随机 `[Mask]` 的 Caption1 | 原始 Caption1 |
| **长描述** | `caption1` 存在 | 图像 + "描述图像" | Caption1 |
| **短描述** | `caption2` 存在 | 图像 + "描述图像" | Caption2 |

---

### 📝 JSON 示例

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

### 3. `Metallogenic Model Diagram – Training/` — 训练图像

存放用于模型训练的成矿模式图，共 **156 张**，文件命名从 `001` 到 `156`，格式涵盖 `.jpg` / `.jpeg` / `.png`。这些图像即为 `dataset.json` 中 `image_path` 字段所指向的视觉输入来源。

### 4. `Metallogenic Model Diagram – Test/` — 测试图像

存放用于模型推理与评估的测试成矿模式图，共 **21 张**，文件命名为 `test_001` 至 `test_021`，格式涵盖 `.png` / `.jpeg`。该目录独立于训练集，用于检验微调后模型的泛化能力。

---

## 🔧 运行环境依赖

训练脚本依赖以下主要库：

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

## 🚀 快速开始

1. 安装上述依赖。
2. 修改训练脚本中的 `dataset_file_path`，指向本项目的 `data/dataset.json`。
3. 修改 `output_dir` 为本地的模型输出目录。
4. 运行对应基座模型的训练脚本：

```bash
python "Code/Qwen2.5-VL-7B – Training.py"
# 或
python "Code/Qwen3-VL-8B – Training.py"
```

5. 训练过程中可通过 SwanLab 面板实时查看 ROUGE / BLEU 等指标曲线。