# Learning Prototypes Under Asymmetric Similarity: Centrality and Stability in Bilinear Classifiers

Isolating the two innovations of the Tversky projection layer — asymmetric similarity and feature-set decomposition — to see which one actually drives its reported gains.

## What this is

The Tversky projection layer ([Doumbouya et al., 2025](#references)) replaces the dot product of a linear classifier with a differentiable version of Tversky's contrast model, and reports accuracy gains on several vision tasks. It changes two things at once relative to a linear layer:

1. it makes the similarity function **asymmetric**, and
2. it represents objects through a learned **feature bank** that supports set-theoretic intersection and difference.

The reported gains could come from either ingredient, or from both. This project separates them. It introduces a prototype classifier whose class score is a bilinear form

```
logit_i = x · M · π_i,    M = S + γA
```

where `S` is symmetric, `A` is skew-symmetric, and `γ` controls the strength of the asymmetry. This gives asymmetry **without** a feature bank, so the asymmetry can be studied on its own and compared against both a plain linear baseline and the full Tversky layer under identical conditions.

All models are trained as classification heads on **frozen** [Barlow Twins](#references) ResNet-50 embeddings of ImageNet-100, so any difference comes from the head, not the backbone.

## Key result

On ImageNet-100 with a frozen self-supervised backbone, no elaborated similarity function beats a plain linear classifier, and accuracy falls monotonically as asymmetry increases.

| Model | Held-out top-1 (%) | Validation (%) | Parameters |
|---|---|---|---|
| **Linear** | **84.41 ± 0.02** | 86.47 | 0.20M |
| Symmetric bilinear (γ=0) | 83.62 ± 0.33 | 86.12 | 8.6M |
| Asymmetric bilinear (γ=1) | 83.10 ± 0.18 | 85.58 | 8.6M |
| Tversky (\|Φ\|=224) | 82.35 ± 0.25 | 85.10 | 0.66M |
| Cosine prototype | 80.96 ± 0.10 | 83.52 | 0.20M |

Held-out accuracy is the mean ± standard deviation over three seeds, with the epoch selected on a held-out validation split.

This does not contradict the original paper, whose gains were on fine-grained recognition. It marks a condition — coarse classes the backbone already separates well — where a richer similarity function has no room to help and instead overfits.

The more interesting finding is structural. Three further analyses show that as a model relies more on a learned similarity metric:

- its **prototypes drift away from their own class members** (centrality margin falls 0.39 → 0.16 → 0.12 → 0.05 across cosine, symmetric, asymmetric, Tversky), and
- its **prototypes become less reproducible across seeds** (cross-seed stability 0.21 for asymmetric, 0.045 for Tversky, against a random-vector floor near 0.02).

The work of classification is off-loaded from the prototypes onto the metric. A representational similarity analysis adds that the asymmetric and Tversky models, despite different mechanisms, converge on nearly the same output geometry (ρ = 0.82 in logit space).

## Repository structure

```
tversky-asymmetric/
├── data/            # ImageNet-100 (not tracked; see Setup)
├── embeddings/      # precomputed frozen-backbone embeddings (.pt)
├── notebooks/       # the experiment pipeline, run in order
├── src/             # reusable code (dataset, models, training)
├── checkpoints/     # trained classification heads (.pt)
├── results/         # figures and metrics
└── report/          # the write-up (LaTeX + PDF)
```

| Notebook | Purpose |
|---|---|
| `01_data_exploration.ipynb` | Load `Labels.json`, build the file manifest, verify class indices |
| `02_embedding_extraction.ipynb` | Extract and cache frozen Barlow Twins embeddings |
| `03_linear_baseline.ipynb` | Linear classification head |
| `04_cosine_prototype.ipynb` | Cosine prototype classifier |
| `05_proposed_model.ipynb` | Asymmetric bilinear classifier (`M = S + γA`) |
| `06_tversky.ipynb` | Tversky projection layer |
| `07_ablation.ipynb` | γ × λ sweep and weight-decay sweep |
| `08_analysis.ipynb` | RSA, prototype centrality, cross-seed stability |

## Setup

The project was developed on Apple Silicon (M3) using the MPS backend, but nothing is platform-specific beyond device selection.

```bash
conda create -n tversky python=3.11
conda activate tversky
pip install torch torchvision numpy scipy scikit-learn matplotlib jupyterlab
```

### Data

The dataset is [ImageNet-100](https://www.kaggle.com/datasets/ambityga/imagenet100) (`ambityga/imagenet100` on Kaggle, ~16 GB). Download it and place it under `data/`. It ships as four training splits (`train.X1`–`train.X4`, 25 classes each) plus a held-out `val.X` (100 classes), with `Labels.json` mapping synsets to class names.

### Backbone

The frozen backbone is a Barlow Twins pretrained ResNet-50, loaded via `torch.hub`:

```python
import torch
backbone = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
backbone.fc = torch.nn.Identity()   # 2048-d embeddings
backbone.eval()
for p in backbone.parameters():
    p.requires_grad = False
```

Preprocessing is the standard `Resize(256) → CenterCrop(224) → Normalize` pipeline with ImageNet channel statistics.

## Reproducing the results

Run the notebooks in order. `02` caches embeddings to `embeddings/` so the later notebooks train on stored vectors rather than re-running the backbone. The train/validation split is stratified with a fixed seed, so the partition is identical across all runs; the official `val.X` is held out and used only for final evaluation.

## The write-up

The full report is in `report/`. It is a NeurIPS-style paper covering the method, the model comparison, the ablations, and the three representational analyses, with a code appendix. Build it with:

```bash
cd report
pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex
```

(`report.tex` needs `references.bib`, `neuripsish.sty`, and the `figures/` directory alongside it.)

## References

- M. K. B. Doumbouya, D. Jurafsky, C. D. Manning. *Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity.* 2025.
- M. Attarian, B. D. Roads, M. C. Mozer. *Transforming Neural Network Visual Representations to Predict Human Judgments of Similarity.* SVRHM Workshop, NeurIPS 2020.
- A. Tversky. *Features of Similarity.* Psychological Review, 1977.
- J. Snell, K. Swersky, R. Zemel. *Prototypical Networks for Few-shot Learning.* NeurIPS 2017.
- C. Chen, O. Li, C. Tao, A. J. Barnett, J. K. Su, C. Rudin. *This Looks Like That: Deep Learning for Interpretable Image Recognition.* NeurIPS 2019.
- E. Rosch. *Cognitive Representations of Semantic Categories.* Journal of Experimental Psychology: General, 1975.
- J. Zbontar, L. Jing, I. Misra, Y. LeCun, S. Deny. *Barlow Twins: Self-Supervised Learning via Redundancy Reduction.* ICML 2021.
