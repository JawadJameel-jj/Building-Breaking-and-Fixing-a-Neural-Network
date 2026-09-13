# Building, Breaking and Fixing a Neural Network

End-to-end feedforward neural network study on Fashion-MNIST: a from-scratch NumPy
implementation, a PyTorch baseline pushed deliberately into overfitting, and a repair pass
using the regularisation and hyperparameter-tuning methods covered in class. Every design
choice is measured against a validation/test metric rather than assumed.

## Contents

| File | Description |
|---|---|
| `neural_network_assignment.ipynb` | The full notebook — all 7 parts, executed top to bottom. |
| `one_page_summary.docx` | One-page summary: final test score, winning configuration, and the single change that helped most. |
| `README.md` | This file. |

## What's inside the notebook

1. **Environment setup** — load Fashion-MNIST, normalise to [0, 1], flatten to 784-dim vectors, 80/20 train/validation split, class distribution report.
2. **Part 1 — Backpropagation from scratch.** A 784→64→10 MLP implemented in raw NumPy (forward, backward, manual gradient descent), trained on a 5000-sample subset, then gradient-checked against an identically-initialised PyTorch model.
3. **Part 2 — Baseline and activation study.** A deeper PyTorch MLP trained once each with sigmoid, tanh, ReLU and leaky ReLU; validation loss curves, first-layer gradient magnitudes, and a dead-ReLU-unit measurement.
4. **Part 3 — Loss functions.** Cross-entropy vs. MSE on one-hot targets for the same classifier, plus a small MLP regression baseline on the scikit-learn diabetes dataset (MSE/RMSE/MAE) — chosen because it ships with scikit-learn and needs no internet access, unlike `fetch_california_housing()`, which Kaggle GPU sessions block by default.
5. **Part 4 — Optimiser comparison.** SGD, SGD+momentum, RMSProp and Adam at a shared learning rate and at per-optimiser tuned rates, with epochs-to-85%, final accuracy and wall-clock time tabulated.
6. **Part 5 — Forcing overfitting.** Training set cut to 2000 samples, network widened to four 512-unit hidden layers, trained past 99% training accuracy; the train/validation separation point and generalisation gap are reported.
7. **Part 6 — Regularisation study.** L2 (3 λ values), L1 (with weight-sparsity %), dropout (3 rates), batch normalisation, early stopping, data augmentation (flip + rotation, vectorised on GPU), and more training data (10k/20k samples) — one comparison table plus gap-vs-strength plots for L2 and dropout. Every run in this section uses early stopping by default so a config that's already converged doesn't keep training for the full epoch budget.
8. **Part 7 — Hyperparameter tuning.** 12-configuration random search over learning rate, hidden width and dropout rate, scored by 5-fold cross-validation; the winning configuration is retrained on the full training set with the best Part 6 regularisation and evaluated once on the untouched test set (accuracy, macro precision/recall/F1, confusion matrix).

## How to reproduce

### 1. Platform
Run on **Kaggle** with the **GPU T4 x2** accelerator enabled (Notebook Settings → Accelerator).

### 2. Add the dataset
Attach the Kaggle dataset **`zalando-research/fashionmnist`** to the notebook
(Add Data → search "Fashion MNIST" → select the Zalando Research version).

The data-loading cell searches `/kaggle/input/**` recursively for
`fashion-mnist_train.csv` / `fashion-mnist_test.csv`, so it finds the files regardless of the
exact folder name Kaggle mounts them under. If the dataset isn't attached at all, that cell
prints the current contents of `/kaggle/input` and then automatically falls back to
`kagglehub.dataset_download("zalando-research/fashionmnist")` to fetch a local copy — no path
edits should be needed either way.

### 3. Upload the notebook
Upload `neural_network_assignment.ipynb` to a new Kaggle notebook (File → Import Notebook),
or copy its cells into a fresh Kaggle notebook.

### 4. Run all cells, top to bottom
`Run All`. All random seeds are fixed (`SEED = 42`) for reproducibility. Approximate runtime
on a single T4 is 20–40 minutes, dominated by Part 6 (12 regularisation runs, each with early
stopping) and Part 7 (5-fold CV × 12 configurations). If Part 6 seems to be taking much longer
than that, check that `early_stopping_patience` wasn't removed from `run_and_record` — without
it, several runs will train for the full epoch budget even after converging.

### 5. Collect the outputs
- All plots render inline and are already labelled per the rubric.
- The Part 4 and Part 6 summary tables print as pandas DataFrames.
- The Part 1 gradient-check values, Part 7 test metrics, and the percentage-point improvement
  over the Part 2/3 baseline print near the end of the notebook. `one_page_summary.docx` is
  already filled in with the results from one such run (see "Results from a completed run"
  below) — if you re-run the notebook, update that document with your own printed values,
  since minor stochastic variation means your numbers may not match exactly.

### Dependencies
Everything needed (`numpy`, `pandas`, `matplotlib`, `torch`, `scikit-learn`, `kagglehub`) is
preinstalled in the standard Kaggle Python GPU environment — no extra `pip install` steps
are required.

## Results from a completed run

The numbers below are from one full run and are already reflected in `one_page_summary.docx`;
your own run may differ slightly since some steps (data augmentation angles, weight init,
minibatch order) are stochastic even with fixed seeds across PyTorch versions.

| Metric | Value |
|---|---|
| Part 2/3 baseline test accuracy | 89.69% |
| Part 7 tuned model test accuracy | 88.66% |
| Macro precision / recall / F1 | 88.73% / 88.66% / 88.67% |
| Improvement over baseline | −1.03 percentage points (tuned model did not beat baseline) |
| Part 7 selected configuration | lr=0.001, hidden width=256, dropout=0.3 (CV search); final retrain used dropout=0.5 per the Part 6 finding |
| Largest gap reduction, Part 6 | More training data (n=10,000): gap fell from 9.23% to 5.59% with no loss in training accuracy |

The negative Part 7 result is reported as-is rather than adjusted — see the one-page summary
for the explanation (short early-stopping patience plus dropout 0.5 likely over-regularised a
model that didn't need as much correction once trained on the full dataset, unlike the
deliberately-overfit 2,000-sample model in Part 6).

## Notes on grading criteria

- Every plot has axis labels and a title.
- Random seeds are fixed in every training loop (`torch.manual_seed`, `np.random.seed`).
- The test set (`X_test` / `y_test`) is loaded once and only used in Part 3 (final baseline
  test accuracy) and Part 7 (final evaluation) — never for model selection, which is done via
  cross-validation on the training data only.
- The Part 7 conclusion is reported honestly: if the tuned model does not beat the baseline,
  the notebook prints and states that directly rather than adjusting the comparison.
