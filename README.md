# Mini Research Problem: Exploring LoRA for Math Reasoning with SFT and GRPO

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Transformers](https://img.shields.io/badge/Transformers-LLM-orange)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-purple)

## Table of Contents

- [Overview](#overview)
- [Research Question](#research-question)
- [Motivation](#motivation)
- [Methods](#methods)
  - [Base Model](#base-model)
  - [Parameter-Efficient Adaptation Background](#parameter-efficient-adaptation-background)
  - [SFT + LoRA](#sft--lora)
  - [GRPO + LoRA](#grpo--lora)
- [Evaluation Design](#evaluation-design)
- [Project Structure](#project-structure)
- [Main Notebook](#main-notebook)
- [Visual Analysis](#visual-analysis)
- [Key Findings](#key-findings)
- [Future Work](#future-work)
- [Reproducibility Notes](#reproducibility-notes)
- [How to Run](#how-to-run)
- [References](#references)
- [Author](#author)

---

## Overview

This mini research problem explores whether **parameter-efficient fine-tuning** can improve mathematical reasoning in a compact local setup.

The project is inspired by the paper **“Learning to Reason in 13 Parameters”** and studies how far reasoning adaptation can be pushed with **small trainable parameter budgets**, rather than full-model fine-tuning.

The main comparison is built on **Qwen/Qwen2.5-Math-1.5B** and includes two post-training paths:

- **SFT + LoRA** with a rank sweep: `r = 8, 4, 2`
- **GRPO + LoRA** with `r = 2`

The goal is to compare these methods on reasoning-oriented math benchmarks and analyze tradeoffs in:

- final answer accuracy
- training cost
- trainable parameter count
- response length
- run-to-run consistency
- parameter efficiency

---

## Research Question

This project asks:

1. Can **small LoRA adapters** recover meaningful gains in mathematical reasoning in a local environment?
2. How does **SFT + LoRA** compare with **GRPO + LoRA** under a low-rank setup?
3. Can this evaluation setup motivate future exploration toward even smaller methods such as **LoRA-XS** and **TinyLoRA**?

---

## Motivation

Large language models can solve many mathematical reasoning tasks, but full fine-tuning is expensive in both compute and memory.

This raises a practical research question:

> Can reasoning ability be improved with a very small number of trainable parameters?

This project focuses on a lightweight and reproducible comparison of parameter-efficient post-training strategies for math reasoning. It is especially motivated by recent work suggesting that reasoning gains may still be possible under extremely small adaptation budgets.

---

## Methods

### Base Model

- `Qwen/Qwen2.5-Math-1.5B`

### Parameter-Efficient Adaptation Background

This project mainly implements **LoRA**, while also positioning **LoRA-XS** and **TinyLoRA** as related methods inspired by *Learning to Reason in 13 Parameters*.

## LoRA

LoRA (**Low-Rank Adaptation**) keeps the original model weights frozen and learns a low-rank update:

$$
W' = W + AB
$$

where $W$ is the frozen base weight, and $A$ and $B$ are trainable low-rank matrices.

Intuitively, LoRA adapts the model by learning a small correction to the original weights instead of updating the full parameter matrix. This makes post-training much cheaper than full fine-tuning, while still allowing meaningful adaptation.

## LoRA-XS

LoRA-XS further reduces the number of trainable parameters by using fixed truncated SVD factors from the base weight matrix:

$$
W' = W + U \Sigma R V^\top
$$

where $U$, $\Sigma$, and $V$ come from the truncated SVD of $W$, and only the small matrix $R$ is trainable.

Compared with standard LoRA, LoRA-XS reduces the trainable parameter count more aggressively by learning only how to recombine dominant singular directions already present in the base model.

## TinyLoRA

TinyLoRA pushes the compression idea further by replacing the trainable core matrix in LoRA-XS with a tiny projected trainable vector:

$$
W' = W + U \Sigma \left(\sum_{i=1}^{u} v_i P_i\right) V^\top
$$

where $P_i$ are fixed random matrices and $v \in \mathbb{R}^u$ is the only trainable vector.

### SFT + LoRA

Supervised fine-tuning (SFT) is used to adapt the model on math-style supervision while keeping the base model frozen and only training LoRA adapter weights.

Pipeline:

`prompt → LoRA-augmented model → predicted answer → compare to gold answer → compute supervised loss → update only LoRA parameters`

Ranks explored:

- `r = 8`
- `r = 4`
- `r = 2`

### GRPO + LoRA

GRPO is used as a reinforcement-style post-training method with LoRA adapters.

This branch explores whether reward-based training can produce competitive reasoning improvements under a very small trainable budget.

Pipeline:

`prompt → model generates a group of answers → reward and rank them → GRPO loss → update LoRA only`

Rank explored:

- `r = 2`

---

## Evaluation Design

The project evaluates each method on three held-out benchmark groups:

- **Reasoning-100**
- **Olympiad-100**
- **AIMO-100**

Each example is evaluated with a **shared protocol**:

- same prompt format
- deterministic decoding
- same answer extraction logic
- same first-line evaluation rule
- multiple runs per example for stability analysis

### Main Metrics

- **Final answer accuracy**
- **Accuracy gain over base**
- **Average generated length**
- **Consistency across 3 runs**
- **Training time**
- **Trainable parameters**
- **Trainable percent**
- **Accuracy gain per 1M trainable parameters**
- **Accuracy per training hour**

---

## Project Structure

```bash
.
├── exploring_lora_fine_tuning_math_reasoning_models_mrp_wo_instruct.ipynb
├── RL+LoRa CPU.ipynb
├── combined_sft_grpo_lora_mrp_integrated.ipynb
├── adapters/
│   ├── qwen_math_lora_r8_300/
│   ├── qwen_math_lora_r4_300/
│   └── qwen_math_lora_r2_300/
├── grpo_lora_qwen15b_math_cpu_out/
│   └── checkpoint-120/
├── data/
│   ├── benchmark_100.jsonl
│   ├── olympiad_100.jsonl
│   └── aimo_100.jsonl
├── mrp_outputs/
└── results_tables/
```

## Main Notebook

The main integrated notebook for this project is:

- `combined_sft_grpo_lora_mrp_integrated.ipynb`

It includes:

1. project motivation and setup  
2. model loading for base / SFT / GRPO  
3. unified evaluation pipeline  
4. benchmark summaries  
5. consolidated visualization section  
6. future exploration notes on LoRA-XS and TinyLoRA  

---

## Visual Analysis

The notebook includes a consolidated visualization section with plots such as:

- accuracy across benchmarks
- consistency across runs
- average token length
- accuracy gain over base
- trainable parameters vs accuracy
- training time vs accuracy
- parameter efficiency plots
- heatmaps of benchmark performance
- SFT rank sweep analysis
- direct comparison of **SFT r=2 vs GRPO r=2**

These visuals are used to compare both **performance** and **efficiency**, not just raw accuracy.

---

## Key Findings

> Fill in this section after running the final notebook.

### Example result summary template

- **Best overall method:** `TODO`
- **Best benchmark on Reasoning-100:** `TODO`
- **Best benchmark on Olympiad-100:** `TODO`
- **Best benchmark on AIMO-100:** `TODO`
- **Most parameter-efficient method:** `TODO`
- **Most time-efficient method:** `TODO`

### Example interpretation template

- SFT with larger LoRA rank may improve raw benchmark performance, but also increases trainable parameter count.
- GRPO + LoRA may offer a different tradeoff between parameter efficiency and raw accuracy.
- Very small adapters remain interesting for future study because they test how far reasoning adaptation can go under tight parameter budgets.

---

## Future Work

This mini research problem outlines two plausible next directions:

### LoRA-XS

A more aggressive low-parameter adaptation strategy that may preserve some reasoning gains while reducing trainable parameters further.

### TinyLoRA

An even smaller adaptation design inspired by ultra-compact reasoning work, potentially useful for studying the lower bound of trainable-parameter reasoning improvement.

These are included as **future exploration paths**, not as completed experimental results.

---

## Reproducibility Notes

To ensure fair comparison, all methods should be evaluated under the same protocol:

- same prompt template
- deterministic decoding
- same token limit
- same first-line extraction rule
- same final answer parser
- same multi-run evaluation procedure

A mismatch in prompt style, decoding length, or extraction logic can change both:

- reported **accuracy**
- reported **response length**

So protocol consistency is critical in this project.

---

## How to Run

### 1. Prepare the environment

Install dependencies with: `pip install transformers peft pandas numpy matplotlib`

If you use the MLX path for SFT evaluation, also install the MLX-based dependencies from the original notebook.

### 2. Make sure model and adapter paths are available

Required resources include:

- Base model: `Qwen/Qwen2.5-Math-1.5B`
- SFT adapters:
  - `adapters/qwen_math_lora_r8_300`
  - `adapters/qwen_math_lora_r4_300`
  - `adapters/qwen_math_lora_r2_300`
- GRPO adapter checkpoint:
  - `grpo_lora_qwen15b_math_cpu_out/checkpoint-120`

### 3. Prepare datasets

Expected benchmark files include:

- `data/benchmark_100.jsonl`
- `data/olympiad_100.jsonl`
- `data/aimo_100.jsonl`

### 4. Run the integrated notebook

Open and run: `combined_sft_grpo_lora_mrp_integrated.ipynb`

### 5. Export outputs

The notebook saves summary tables such as:

- `all_summaries_sft.csv`
- `all_summaries_grpo.csv`
- `all_compare.csv`
- ranking tables and benchmark summaries

---

## References

- **Qwen/Qwen2.5-Math-1.5B**
- **LoRA: Low-Rank Adaptation of Large Language Models**
- **GRPO / reinforcement-style post-training methods**
- **Learning to Reason in 13 Parameters**

---

## Author

**Lusha Zhang**

This project was developed as a mini research problem, focusing on parameter-efficient post-training for mathematical reasoning with compact local experimentation.
