# LLM Fine-Tuning Showdown: LoRA vs QLoRA vs Full Fine-Tuning

**Research Question**: Which LLM fine-tuning method provides the best trade-off between accuracy, cost, and resource requirements for practical applications?

## 🎯 Objective

Compare three fine-tuning methods on a resume classification task:
- **Full Fine-Tuning**: Update all model parameters
- **LoRA**: Low-Rank Adaptation (parameter-efficient)
- **QLoRA**: Quantized LoRA (memory-efficient)

## 🏆 Key Results

**Hardware**: NVIDIA A100-SXM4-80GB (Google Colab Pro+)

**Dataset**: 962 resumes across 25 job categories (70/15/15 split)

**Base Model**: mistralai/Mistral-7B-v0.1

### Performance Summary:

```
============================================================
FINAL COMPARISON
============================================================
Baseline:     73.00% | 0h training
Full FT:      100.00% | 0.20h training (12 min)
LoRA:         93.79% | 0.06h training (3.7 min) - FASTEST ⚡
QLoRA:        94.48% | 0.10h training (6 min) - BEST EFFICIENCY 🎯
============================================================
```

| Method | Accuracy | F1-Score | Training Time | GPU Memory | Trainable Params |
|--------|----------|----------|---------------|------------|------------------|
| **Baseline** | 73.00% | 0.7426 | 46s | N/A | N/A |
| **Full FT** | **100.00%** 🥇 | **1.0000** | 12 min | ~60 GB (75%) | 7.11B (100%) |
| **QLoRA** | **94.48%** 🥈 | **0.9392** | 6 min | **~15 GB (17%)** 🎯 | 3.5M (0.05%) |
| **LoRA** | **93.79%** 🥉 | **0.9306** | 3.7 min | ~80 GB (98%) ⚡ | 3.5M (0.05%) |

### Key Findings:

1. ✅ **Full FT**: Perfect 100% accuracy - Gold standard, but expensive (60GB GPU, 7.11B params)
2. ✅ **QLoRA**: 94.48% accuracy with MAXIMUM memory efficiency (~15GB, 17% util) - **Best for production!** 🎯
3. ✅ **LoRA**: Fastest training (3.7 min), but uses 80GB GPU memory (98% util) - more than Full FT!
4. ✅ **Surprising Finding**: LoRA uses MORE memory than Full FT despite training only 0.05% of params!
5. ✅ **Winner**: QLoRA strikes the best balance - beats LoRA accuracy with 80% less GPU memory!

## 📊 Task

**Resume Classification**: Categorize resumes into 25 job categories

**Dataset**: UpdatedResumeDataSet.csv from Kaggle (962 samples)

**Evaluation Metrics**:
- Accuracy & F1-score ✅
- Training time ✅
- GPU memory usage ✅
- Trainable parameters ✅
- Model size ✅

## 🗂️ Project Structure

```
llm-finetuning-research/
├── data/
│   ├── raw/              # Kaggle resume dataset
│   └── processed/        # Train/val/test splits
├── src/
│   ├── data/            # Data processing scripts
│   ├── training/        # Training scripts for each method
│   └── evaluation/      # Evaluation and analysis
├── models/              # Saved models and checkpoints
├── experiments/results/ # Experimental results and logs
├── notebooks/           # Analysis notebooks
└── papers/drafts/       # arXiv paper drafts
```

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download dataset** (from Kaggle):
   - Dataset: Resume Dataset or similar
   - Place in `data/raw/`

3. **Process data**:
   ```bash
   python src/data/prepare_dataset.py
   ```

4. **Run experiments**:
   ```bash
   # Full fine-tuning
   python src/training/full_finetune.py

   # LoRA
   python src/training/lora_finetune.py

   # QLoRA
   python src/training/qlora_finetune.py
   ```

5. **Evaluate**:
   ```bash
   python src/evaluation/evaluate_model.py
   ```

## 📊 Weights & Biases Tracking

All experiments tracked with W&B:

- **Project**: [llm-finetuning-showdown](https://wandb.ai/birla2006-independent-researcher/llm-finetuning-showdown)
- **Baseline Run**: [sz78gpo9](https://wandb.ai/birla2006-independent-researcher/llm-finetuning-showdown/runs/sz78gpo9)
- **Full FT Run**: [e45zpfah](https://wandb.ai/birla2006-independent-researcher/llm-finetuning-showdown/runs/e45zpfah)
- **LoRA Run**: [wia3xlss](https://wandb.ai/birla2006-independent-researcher/llm-finetuning-showdown/runs/wia3xlss)
- **QLoRA Run**: [9gaihret](https://wandb.ai/birla2006-independent-researcher/llm-finetuning-showdown/runs/9gaihret)

**📄 W&B Report**: Comprehensive 5-page PDF report with all visualizations available at `wandb_report.pdf` - includes:
- Accuracy, F1, precision, recall comparison charts
- Training curves and convergence analysis
- GPU utilization and system metrics
- Final performance comparison across all methods

## 📁 Notebooks

All training notebooks are in the `notebooks/` directory:
- `Setup_and_Baseline.ipynb` - Zero-shot baseline
- `Full_FineTuning.ipynb` - Full fine-tuning (100% accuracy)
- `LoRA_FineTuning.ipynb` - LoRA training (93.79% accuracy)
- `QLoRA_FineTuning.ipynb` - QLoRA training (94.48% accuracy)

**Hardware**: Google Colab Pro+ with A100-SXM4-80GB GPU


## 📧 Contact

**Researcher**: Birla Murugesan
**Institution**: Independent Researcher
**Project**: LLM Fine-Tuning Comparative Study
**Date**: December 2025
