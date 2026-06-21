# Module 37: Graded Mini Project - Conditional GAN

This project implements a Conditional Generative Adversarial Network (cGAN) for the TensorFlow Datasets `cats_vs_dogs` dataset.

Assignment label convention:

| Label | Class |
|---:|---|
| 0 | Cat |
| 1 | Dog |

## Files

- `src/cgan_cats_vs_dogs.py` - full reusable training implementation.
- `notebooks/Module_37_Graded_Mini_Project_Manoj_Bhardwaj.ipynb` - generated notebook-style submission artifact.
- `scripts/build_submission_artifacts.py` - builds the notebook and final PDF report.
- `outputs/samples/` - generated real and fake image grids after training.
- `outputs/models/` - saved Keras generator and discriminator models.
- `Module 37 - Graded Mini Project_Manoj_Bhardwaj.pdf` - final PDF report artifact.

## Recommended Environment

Use Python 3.11 for the TensorFlow runtime on Windows.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run A Fast Architecture Check

This checks model shapes and one synthetic train step without downloading the dataset.

```powershell
python src\cgan_cats_vs_dogs.py --architecture-check
```

## Run A Dataset Smoke Test

This checks the pipeline with one epoch and one batch.

```powershell
python src\cgan_cats_vs_dogs.py --smoke-test
```

If Windows/Python reports an SSL certificate verification error while TFDS downloads
`cats_vs_dogs`, use a local TFDS cache and disable SSL verification only for the
dataset download:

```powershell
python src\cgan_cats_vs_dogs.py --smoke-test --batch-size 8 --data-dir .tfds --disable-download-ssl-verification
```

## Run The Required Training

The assignment requires at least 10 epochs.

```powershell
python src\cgan_cats_vs_dogs.py --epochs 10 --batch-size 128
```

Use the same SSL flag for the full run if the dataset download fails with the
certificate error:

```powershell
python src\cgan_cats_vs_dogs.py --epochs 10 --batch-size 128 --data-dir .tfds --disable-download-ssl-verification
```

The script saves:

- `outputs/samples/real_preprocessed_samples.png`
- `outputs/samples/epoch_000_untrained.png`
- `outputs/samples/epoch_001.png` through `epoch_010.png`
- `outputs/models/conditional_generator.keras`
- `outputs/models/conditional_discriminator.keras`

## Rubric Alignment

The implementation explicitly covers dataset loading, preprocessing, conditional generator, conditional discriminator, separate loss functions, separate Adam optimizers, an optimized `@tf.function` training step, output sampling, and interpretation guidance.
