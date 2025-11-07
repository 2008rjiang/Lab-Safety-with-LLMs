#Laboratory Safety Dataset and Multimodal Evaluation Framework

This repository contains code, data samples, and evaluation scripts from our study:  
“Evaluating Multimodal Large Language Models for Real-World Laboratory Safety Reasoning.”

The project introduces the first large-scale dataset of authentic laboratory images annotated for safety compliance, laboratory type, and violation categories. It further benchmarks a range of multimodal large language models (MLLMs) in hazard detection and reasoning fidelity.

---

## Dataset Overview

The dataset comprises 1,527 authentic laboratory images, curated primarily from LAION-400M and filtered via a CLIP-based semantic retrieval pipeline.  
Each image is annotated with structured metadata capturing:

- **`lab_type`**: `Biology`, `Chemistry`, or `Electrical Engineering`
- **`unsafe`**: Binary safety label (0 = safe, 1 = unsafe)
- **`category`**: Violation type (`PPE`, `SOP`, or `WO`)
- **`details`**: Human-readable description of violations
- **`gpt_reason`**: Expert-style textual reasoning generated via LLM prompts

### Dataset Split

| Subset | # Images | Safe | Unsafe | Description |
|:--|--:|--:|--:|:--|
| Train | 1,218 | 606 | 612 | Used for model fine-tuning and prompt optimization |
| Test | 309 | 152 | 157 | Reserved for benchmarking; high-certainty labels only |

Splits were generated using stratified sampling to preserve distribution across:
- Safety status (`safe` / `unsafe`)
- Laboratory domain (`biology`, `chemistry`, `ee`)
- Violation category (`ppe`, `sop`, `wo`)

Random seed fixed at `42` for reproducibility.

---

## Annotation Examples

Each entry includes a structured label schema and reasoning text.  
Below are two representative examples:

| Image | Lab Type | Unsafe | Category | Description |
|:--|:--|:--|:--|:--|
| ![Example 1](examples/image0070.jpg) | Biology | 0 | PPE | “Individual wears mask, gloves, and gown; workspace organized and sterile — no evident hazards.” |
| ![Example 2](examples/image1267.jpg) | Chemistry | 1 | SOP | “Missing gloves; distillation running in open air; unsafe proximity to apparatus — overall unsafe.” |

(Images shown here are sample thumbnails; full dataset images are excluded for licensing reasons.)

---

## Evaluation Framework

Each model was prompted with the following evaluation instruction:

> *“You are a laboratory safety expert. Analyze the image carefully and list all visible safety violations. Provide reasoning for each and conclude whether the lab scene is safe or unsafe.”*

### Models Evaluated

| Model | Type | Year | Vision Encoder |
|:--|:--|:--:|:--|
| GPT-4o | Proprietary | 2024 | Internal |
| GPT-4o-mini | Proprietary | 2024 | Internal |
| GPT-5-nano | Proprietary | 2025 | Internal |
| Qwen2.5-VL | Open-weight | 2025 | ViT-G |
| LLaVA-1.6 | Open-weight | 2024 | CLIP-L/14 |

Each model’s predictions were evaluated along three axes:

1. **Hazard Detection Accuracy (HDA)** — correctness of safe/unsafe classification  
2. **Violation Categorization (VC)** — accuracy of predicting the correct category (PPE, SOP, WO)  
3. **Reasoning Fidelity (RF)** — factual consistency and absence of hallucinated violations  

Evaluation was performed using **GPT-5 as a reference grader**, following a two-stage pipeline:  
(1) generation → (2) LLM-based structured evaluation.

---

## Results Summary

| Model | HDA (↑) | VC (↑) | RF (↑) | Notes |
|:--|:--:|:--:|:--:|:--|
| GPT-5-nano | **89.7%** | **85.1%** | **91.2%** | Most balanced reasoning and low hallucination rate |
| GPT-4o | 84.5% | 80.6% | 87.0% | Occasional overconfidence on ambiguous images |
| GPT-4o-mini | 79.3% | 76.8% | 81.4% | Reasonable detection but frequent omission errors |
| Qwen2.5-VL | 71.2% | 68.4% | 73.5% | Struggles with multi-object reasoning |
| LLaVA-1.6 | 67.9% | 61.7% | 65.8% | High hallucination rate; poor fine-grained detection |

### Qualitative Case Studies

**Example 1:**  
*Image:* Chemistry lab with open heating mantle and missing gloves.  
- GPT-4o correctly flagged missing gloves but ignored exposed flask.  
- Qwen2.5-VL hallucinated “smoke” and incorrectly labeled it as fire hazard.  
- GPT-5-nano identified both violations and recommended procedural correction.

**Example 2:**  
*Image:* Biology lab with clean workspace.  
- All models classified as safe.  
- GPT-4o-mini added an unnecessary “missing goggles” warning (false positive).  

---
