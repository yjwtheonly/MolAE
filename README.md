# MolAE: Auto-Encoder Based Molecular Representation Learning With 3D Cloze Test Objective

```
Official implementation of paper: MolAE: Auto-Encoder Based Molecular Representation Learning With 3D Cloze Test Objective (icml 2024)
```

The code of Mol-AE 主要基于 UniMol codebase 来构建, 因此，相关环境的配置请参照 UniMol。

## Pre-train
There are two ways to use Mol-AE, you can 直接使用我们预训练好的权重，或者从头开始预训练 Mol-AE。
### 1.Directly use the pre-trained model.
The pre-trained checkpoint can be downloaded from [google drive] (https://drive.google.com/file/d/1NKObZCfE80GCLS9yJ7hqMGzjfGol4LLo/view?usp=drive_link).

### 2. Pre-train Mol-AE from scratch.

## Downstream Tasks
我们使用了和UniMol完全一致的下游任务，主要包含两大类的任务：classification and regression.

### Classification

### Regression