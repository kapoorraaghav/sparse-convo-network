# Submanifold Sparse CNNs for Particle Jet Classification

A PyTorch implementation of Submanifold Sparse Convolutional Networks 
(Graham & van der Maaten, 2017) applied to particle jet classification, 
developed as part of a GSOC proposal for ML4Sci.

---

## Overview

Particle jet images are naturally sparse — most detector cells are empty. 
Standard CNNs waste compute processing these empty regions. This project 
implements Submanifold Sparse Convolutions (VSC) which skip inactive sites, 
reducing memory and computation while preserving accuracy.

---

## Implementation

Two core sparse operators implemented from scratch in PyTorch (`model.py`):

- **SubmanifoldSparseConv2d (VSC)** — preserves sparsity mask across layers 
  using ground state subtraction (Section 2, Graham & van der Maaten 2017)
- **StridedSparseConv2d (SC)** — downsampling while expanding active sites

---

## Pipeline

**Stage 1 — Unsupervised Pretraining** (`model_1.ipynb`)
- Sparse autoencoder trained on 60,000 unlabelled jet images (125×125, 8 channels)
- Reconstruction loss computed at active sites only
- SGD (lr=1e-3, momentum=0.9, weight_decay=1e-4)

**Stage 2 — Fine-tuning** (`finetuning.ipynb`)
- Sparse classifier fine-tuned on 10,000 labelled samples
- Two-phase training: frozen encoder (10 epochs) → full fine-tuning
- SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)

```

## Results

### Classification
| Setting                        | Val Accuracy |  |AUC|
| Initial (Adam + dropout 0.9)   | ~50% |
| After fixes (SGD, no dropout)  | **81.75%** |   |0.89|

### Dense Baseline Comparison
| Metric        | Dense   | Sparse | Ratio              |

| Active States | 13,107K | 6,513K | **2.0x reduction** |

*Note: FLOPs comparison requires a native sparse backend (spconv). 
Implementing true sparse FLOPs reduction is the proposed GSOC contribution.*
```

## Project Structure
~~~
├-- model.py                          # Sparse conv operators + autoencoder
├-- model_1.ipynb                     # Unsupervised pretraining
├-- finetuning.ipynb                  # Fine-tuning + evaluation
|-- dense.ipynb
|-- comparison.ipynb                  # shows comaparison between dense and sparse   baseline comparison
|-- pruning.ipynb  
├-- models/
│   -- sparse_autoencoder_checkpoint_2.pth
│   -- sparse_classifier.pth
|   -- dense.pth
└── dataset/                          # Not included

~~~
## Known Limitations

Current implementation uses standard PyTorch Conv2d with explicit masking 
rather than a true sparse CUDA backend. This means:
- Active state reduction is real and validated 
- FLOPs are not genuinely reduced yet 

Replacing masking with spconv for true computational savings is the 
core proposed GSOC contribution.



## Reference

Graham, B., & van der Maaten, L. (2017). 
*Submanifold Sparse Convolutional Networks.* 
arXiv:1706.01307
