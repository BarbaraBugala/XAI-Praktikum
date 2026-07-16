# XAI-Praktikum

This repository compares three Shapley-value attribution methods for image classification:

- **Shapiq Vision**
- **ProxySHAP**
- **Captum KernelSHAP**

The comparison is organized into three notebooks that progressively move from a single-image demonstration to robustness analysis and finally to a large-scale quantitative benchmark.

## Repository Overview

| Notebook | Purpose |
|----------|---------|
| `comparison_of_methods.ipynb` | Side-by-side comparison of all three methods on a single ImageNet image |
| `shapiq_vision_noise_test.ipynb` | Analysis of attribution stability under increasing image noise |
| `faithfulnessAID_benchmark.ipynb` | Large-scale benchmark using the insertion/deletion AID faithfulness metric |

---

## Notebooks

### `comparison_of_methods.ipynb`

This notebook serves as the main introduction to the repository. It compares all three attribution methods on a single image using a ResNet-18 classifier.

![Guitardog](data/guitardog.png "Guitar dog image")

The chosen image is an interesting stress test because it contains two valid semantic concepts-a **golden retriever** and a **guitar**. We explain two different model predictions: the top-1 prediction and the model's fifth-ranked class.

The notebook includes:

- an introduction to Shapley values,
- conceptual explanations of each attribution method,
- mathematical derivations of Kernel SHAP and ProxySHAP,
- side-by-side superpixel attribution heatmaps,
- bar plots showing the ten most important superpixels for each method,
- a comparison of runtime and model-call budgets.

---

### `shapiq_vision_noise_test.ipynb`

This notebook investigates the robustness of **Shapiq Vision** under image perturbations.

Starting from the exact setup used in the comparison notebook, Gaussian noise is added to the input image at several increasing noise levels while keeping the segmentation and model unchanged. The resulting attributions are then compared with those obtained from the clean image.

The notebook provides:

- attribution heatmaps for each noise level,
- a qualitative analysis of how explanations change as the image becomes increasingly noisy.

Although the focus is on Shapiq Vision, the experiment also illustrates how the underlying **ResNet-18** model behaves under severe image corruption and how attribution methods can help identify where the model begins to fail.

---

### `faithfulnessAID_benchmark.ipynb`

This notebook extends the comparison from a single example to a quantitative benchmark over approximately **200 ImageNet images**.

Faithfulness is evaluated using the **Area between Insertion and Deletion (AID)** metric, where higher values indicate explanations that better capture the model's decision process.

The current benchmark compares:

- **Shapiq Vision**
- **Captum KernelSHAP**

across

- two segmentation strategies (superpixels and regular grids), and
- two image classification models (ResNet-18 and ViT-B/16).

The notebook is organized into two sections:

### 1. Experiment

Runs the complete benchmark (disabled by default via `RUN_EXPERIMENT = False`).

Results are stored in `aid_sweep_results.pkl`, allowing interrupted runs to resume automatically. Setting `PILOT = True` executes the benchmark on only a small subset of images for quick testing.

### 2. Results

Loads the stored benchmark results and produces:

- insertion and deletion curves,
- per-image AID agreement scatter plots,
- mean AID comparisons across segmentation strategies,
- summary tables,

with all figures shown separately for ResNet-18 and ViT-B/16.

---

## Installation

Install the required packages with:

```bash
pip install "git+https://github.com/S2k-1/shapiq.git@feature/protocols" captum scikit-image --quiet
```

---

## Suggested Reading Order

1. `comparison_of_methods.ipynb` - Learn the intuition behind each attribution method and compare them on a single image.
2. `shapiq_vision_noise_test.ipynb` - Explore how Shapiq Vision explanations behave under increasing image noise.
3. `faithfulnessAID_benchmark.ipynb` - Examine the large-scale quantitative comparison using the AID faithfulness metric.