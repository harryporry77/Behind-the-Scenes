# Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition

Official implementation of the paper "Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition" ([arXiv:2509.08454](https://arxiv.org/abs/2509.08454)).

## Abstract

Large pre-trained speech models such as Whisper offer strong generalization but pose significant challenges for resource-efficient adaptation. Low-Rank Adaptation (LoRA) has become a popular parameter-efficient fine-tuning method, yet its underlying mechanisms in speech tasks remain poorly understood. In this work, we conduct the first systematic mechanistic interpretability study of LoRA within the Whisper encoder for speech emotion recognition (SER). Using a suite of analytical tools, including layer contribution probing, logit-lens inspection, and representational similarity via singular value decomposition (SVD) and centered kernel alignment (CKA), we reveal two key mechanisms: a delayed specialization process that preserves general features in early layers before consolidating task-specific information, and a forward alignment, backward differentiation dynamic between LoRA’s matrices. Our findings clarify how LoRA reshapes encoder hierarchies, providing both empirical insights and a deeper mechanistic understanding for designing efficient and interpretable adaptation strategies in large speech models.


## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training LoRA-adapted Whisper
```bash
python train_lora.py
```

### Running Mechanistic Analysis
```bash
jupyter notebook analysis.ipynb
```

## Citation
```bibtex
@article{li2024behind,
  title={Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition},
  author={Li, Ruizhe and others},
  journal={arXiv preprint arXiv:2509.08454},
  year={2024}
}
```