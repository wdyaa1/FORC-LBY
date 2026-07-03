# FORC Simulation-to-Real Baseline

Baseline scaffold for the project:

**Bridging the Simulation-to-Real Gap in Machine Learning Inversion of FORC Diagrams**

The goal is to train a first supervised model on simulated FORC diagrams, then measure how robust it is under noise, intensity shifts, and, later, experimental FORC data.

## Baseline Scope

This folder gives you a clean starting point for:

- loading simulated FORC diagrams from `.csv` or `.frc` files;
- converting every diagram to a consistent 2D grid;
- normalising FORC density values consistently;
- training a small CNN baseline;
- testing noise and intensity augmentation;
- evaluating simulation accuracy and robustness;
- leaving clear extension points for real experimental data and fine-tuning.

## Suggested Folder Layout

```text
forc_sim2real_baseline/
  configs/
    baseline.yaml
  data/
    raw/
      simulated/
      experimental/
    processed/
  notebooks/
    01_data_inspection.md
  reports/
    figures/
    baseline_results.md
  src/
    forc_baseline/
      data.py
      preprocess.py
      augment.py
      model.py
      train.py
      evaluate.py
      utils.py
  tests/
    test_preprocess.py
  requirements.txt
  run_baseline.ps1
```

## Quick Start

1. Create a Python environment.
2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Put simulated FORC files into:

```text
data/raw/simulated/
```

4. Put real/experimental FORC files into:

```text
data/raw/experimental/
```

5. Run the stage 1 data inspection:

```powershell
.\run_stage1_inspection.ps1
```

This creates:

```text
data/raw/simulated_labels_template.csv
reports/stage1_data_inspection.md
```

6. Fill the label template and save it as:

```text
data/raw/simulated_labels.csv
```

7. Edit `configs/baseline.yaml` so that the label column or label file matches your available simulated data.

8. Run:

```powershell
.\run_baseline.ps1
```

## Stage 2 Smoke Test

Before real training, you can verify that the preprocessing path works:

```powershell
.\run_stage2_smoke_test.ps1
```

This creates a tiny synthetic dataset and writes:

```text
data/raw/simulated_smoke/
data/raw/simulated_smoke_labels.csv
reports/stage2_preprocessing_smoke_test.md
reports/figures/stage2_smoke/
```

The smoke test is only for checking the pipeline. Do not use its synthetic files as scientific results.

## Baseline Experiment

The first experiment should answer:

> How well does a model trained only on numerical FORC data predict grain-size or domain-state targets on unseen simulated data?

Recommended minimum reporting:

- train/validation/test split strategy;
- target definition: regression or classification;
- input grid size and normalisation method;
- MAE/RMSE for regression, or accuracy/F1 for classification;
- robustness under added noise;
- examples of successful and failed predictions.

## Next Extensions

After the clean simulation baseline works:

- add noise augmentation;
- add intensity/scale augmentation;
- compare simulated and experimental FORC statistics;
- fine-tune on a small labelled or expert-interpreted real subset if available;
- add feature-level alignment only if the simpler methods are not enough.
