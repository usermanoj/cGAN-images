"""Build notebook and PDF submission artifacts for the Module 37 cGAN project."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "cgan_cats_vs_dogs.py"
NOTEBOOK_PATH = (
    ROOT / "notebooks" / "Module_37_Graded_Mini_Project_Manoj_Bhardwaj.ipynb"
)
PDF_PATH = ROOT / "Module 37 - Graded Mini Project_Manoj_Bhardwaj.pdf"
SAMPLE_DIR = ROOT / "outputs" / "samples"


def read_source() -> str:
    return SOURCE_PATH.read_text(encoding="utf-8")


def make_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().splitlines()],
    }


def build_notebook() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        make_markdown_cell(
            """
            # Module 37: Graded Mini Project - Conditional GAN

            **Student:** Manoj Bhardwaj

            **Goal:** Build a Conditional Generative Adversarial Network (cGAN) to generate 64x64 RGB images of cats or dogs conditioned on class labels.

            **Required label convention:**

            | Label | Class |
            |---:|---|
            | 0 | Cat |
            | 1 | Dog |
            """
        ),
        make_markdown_cell(
            """
            ## Phase 1 - Data Preparation

            The dataset is loaded from TensorFlow Datasets using `cats_vs_dogs`. Images are resized to 64x64, normalized to `[-1, 1]`, labels are cast to integer tensors, and the training pipeline uses shuffle, batch, and prefetch.
            """
        ),
        make_code_cell(
            """
            import sys
            from pathlib import Path

            PROJECT_ROOT = Path.cwd()
            if not (PROJECT_ROOT / "src").exists() and (PROJECT_ROOT.parent / "src").exists():
                PROJECT_ROOT = PROJECT_ROOT.parent
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            from src.cgan_cats_vs_dogs import (
                CGANConfig,
                build_discriminator,
                build_generator,
                label_names_from_info,
                load_raw_datasets,
                prepare_dataset,
                save_real_sample_grid,
                train_cgan,
            )

            config = CGANConfig(epochs=10, batch_size=128, output_dir=Path("outputs"))
            DATA_DIR = ".tfds"
            VERIFY_SSL = False  # Set True if your local certificates allow TFDS downloads.

            ds_train_raw, ds_test_raw, ds_info = load_raw_datasets(
                data_dir=DATA_DIR,
                verify_ssl=VERIFY_SSL,
            )
            class_names = label_names_from_info(ds_info)
            print(f"Verified label mapping: 0 = {class_names[0]}, 1 = {class_names[1]}")

            train_dataset = prepare_dataset(ds_train_raw, config, training=True)
            test_dataset = prepare_dataset(ds_test_raw, config, training=False)
            save_real_sample_grid(
                test_dataset,
                class_names,
                Path("outputs/samples/real_preprocessed_samples.png"),
                config,
            )
            """
        ),
        make_markdown_cell(
            """
            ## Phase 2 - Conditional GAN Architecture

            The generator accepts random noise and a class label. The label is embedded and concatenated with the noise vector before transposed convolution layers generate a `64x64x3` RGB image.

            The discriminator accepts an image and a class label. The label is embedded into a spatial `64x64x1` map and concatenated with the image before Conv2D layers classify the input as real or fake.
            """
        ),
        make_code_cell(
            """
            generator = build_generator(config)
            discriminator = build_discriminator(config)

            generator.summary()
            discriminator.summary()
            """
        ),
        make_markdown_cell(
            """
            ## Phase 3 and 4 - Losses, Optimizers, and Training

            The implementation uses `BinaryCrossentropy(from_logits=True)`. The generator is trained to make generated images look real to the discriminator. The discriminator is trained to classify real images as real and generated images as fake. Separate Adam optimizers are used for the two networks.
            """
        ),
        make_code_cell(
            """
            history, generator, discriminator = train_cgan(
                config,
                data_dir=DATA_DIR,
                verify_ssl=VERIFY_SSL,
                max_train_batches=None,
                sample_every=1,
            )

            history
            """
        ),
        make_markdown_cell(
            """
            ## Phase 5 - Evaluation and Visualization

            The training script saves a fixed class-conditioned image grid after each epoch. The first half of the grid is conditioned on label `0` (Cat), and the second half is conditioned on label `1` (Dog). Generated tensors are converted from `[-1, 1]` back to `[0, 1]` before display.

            Review these files after the run:

            - `outputs/samples/epoch_000_untrained.png`
            - `outputs/samples/epoch_001.png` through `outputs/samples/epoch_010.png`
            - `outputs/samples/real_preprocessed_samples.png`
            """
        ),
        make_markdown_cell(
            """
            ## Results Interpretation

            After a 10-epoch training run, evaluate:

            - **Image quality:** whether generated samples show recognizable cat/dog-like shapes, colors, and texture.
            - **Diversity:** whether samples differ from each other instead of collapsing to a single repeated pattern.
            - **Label conditioning:** whether outputs conditioned on `Cat` and `Dog` show visible class-specific differences.
            - **Limitations:** a 10-epoch run may produce noisy or partial animals because GANs often need longer training and tuning for high-quality images.
            - **Possible improvements:** train longer, tune learning rate/batch size, increase model capacity, add augmentation, or use more advanced GAN losses.
            """
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(textwrap.dedent(text).strip().replace("\n", "<br/>"), style)


def build_pdf() -> None:
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="SmallCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=6.6,
            leading=8.0,
            alignment=TA_LEFT,
        )
    )
    title_style = ParagraphStyle(
        name="ProjectTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#19324d"),
        spaceAfter=18,
    )
    h1 = ParagraphStyle(
        name="ProjectH1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#19324d"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body = ParagraphStyle(
        name="ProjectBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        spaceAfter=7,
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.65 * inch,
        title="Module 37 - Graded Mini Project",
        author="Manoj Bhardwaj",
    )

    story = []
    story.append(Paragraph("Module 37: Graded Mini Project", title_style))
    story.append(para("<b>Student:</b> Manoj Bhardwaj", body))
    story.append(
        para(
            """
            <b>Project:</b> Conditional Generative Adversarial Network (cGAN) for Cats vs Dogs image generation.
            The model conditions both generator and discriminator on class labels where 0 = Cat and 1 = Dog.
            """,
            body,
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("Project Objective", h1))
    story.append(
        para(
            """
            Build an end-to-end cGAN pipeline that loads the TensorFlow Datasets `cats_vs_dogs` dataset,
            preprocesses the images, defines conditional generator and discriminator networks, trains
            them adversarially for at least 10 epochs, and saves label-conditioned visual outputs.
            """,
            body,
        )
    )

    story.append(Paragraph("Rubric Alignment", h1))
    table_data = [
        ["Criterion", "Implementation Evidence"],
        ["Dataset Loading", "Uses `tfds.load('cats_vs_dogs')` with 90/10 train/test split and verifies 0 = Cat, 1 = Dog."],
        ["Preprocessing", "Resizes to 64x64, normalizes to [-1, 1], casts labels to int32, batches and prefetches."],
        ["Generator", "Noise and label inputs; label embedding; concatenation; Conv2DTranspose upsampling; tanh 64x64x3 output."],
        ["Discriminator", "Image and label inputs; spatial label embedding; image-label concatenation; Conv2D real/fake logit."],
        ["Training", "`@tf.function` train step with separate generator and discriminator gradient updates."],
        ["Loss and Optimizers", "Binary cross-entropy from logits; separate Adam optimizers with learning rate 2e-4 and beta_1 0.5."],
        ["Visualization", "Saves real sample grid, untrained baseline grid, and epoch-wise conditioned generated grids."],
        ["Interpretation", "Notebook includes prompts for image quality, diversity, label conditioning, limitations, and improvements."],
    ]
    table = Table(table_data, colWidths=[1.55 * inch, 5.0 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#19324d")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#aeb8c2")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("Execution Instructions", h1))
    story.append(
        para(
            """
            Recommended Windows runtime: Python 3.11. Create a virtual environment, install `requirements.txt`,
            then run `python src\\cgan_cats_vs_dogs.py --epochs 10 --batch-size 128 --data-dir .tfds`.
            If local certificate verification blocks the TFDS download, add `--disable-download-ssl-verification`.
            For a fast no-download validation, run `python src\\cgan_cats_vs_dogs.py --architecture-check`.
            For a dataset pipeline check, run `python src\\cgan_cats_vs_dogs.py --smoke-test`.
            """,
            body,
        )
    )

    story.append(Paragraph("Output Files", h1))
    story.append(
        para(
            """
            The implementation saves generated grids in `outputs/samples/`, including `real_preprocessed_samples.png`,
            `epoch_000_untrained.png`, and one generated image grid per epoch. The final trained models are saved as
            Keras files in `outputs/models/`.
            """,
            body,
        )
    )

    story.append(Paragraph("Visual Output Samples", h1))
    real_sample = SAMPLE_DIR / "real_preprocessed_samples.png"
    untrained_sample = SAMPLE_DIR / "epoch_000_untrained.png"
    trained_candidates = sorted(
        [
            path
            for path in SAMPLE_DIR.glob("epoch_*.png")
            if path.name != "epoch_000_untrained.png"
        ]
    )
    visual_items = []
    if real_sample.exists():
        visual_items.append(("Real preprocessed validation samples", real_sample))
    if untrained_sample.exists():
        visual_items.append(("Untrained generator baseline", untrained_sample))
    if trained_candidates:
        visual_items.append(
            (
                f"Latest class-conditioned generated grid ({trained_candidates[-1].name})",
                trained_candidates[-1],
            )
        )

    if visual_items:
        for caption, image_path in visual_items:
            story.append(Paragraph(caption, styles["Heading3"]))
            img = Image(str(image_path))
            max_width = 6.2 * inch
            max_height = 4.4 * inch
            scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
            img.drawWidth = img.imageWidth * scale
            img.drawHeight = img.imageHeight * scale
            story.append(img)
            story.append(Spacer(1, 8))
    else:
        story.append(
            para(
                """
                No generated sample PNGs were present when this PDF was built.
                Run the training command, then rerun `python scripts\\build_submission_artifacts.py`
                to embed the latest generated output grid.
                """,
                body,
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Appendix: Complete Python Implementation", h1))
    source = read_source()
    chunks = []
    current = []
    current_len = 0
    for line in source.splitlines():
        if len(line) <= 104:
            wrapped_lines = [line]
        else:
            indent = len(line) - len(line.lstrip(" "))
            continuation_indent = " " * min(indent + 4, 24)
            wrapped_lines = textwrap.wrap(
                line,
                width=104,
                subsequent_indent=continuation_indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        current.extend(wrapped_lines)
        current_len += len(wrapped_lines)
        if current_len >= 72:
            chunks.append("\n".join(current))
            current = []
            current_len = 0
    if current:
        chunks.append("\n".join(current))

    for index, chunk in enumerate(chunks):
        if index:
            story.append(PageBreak())
        story.append(Preformatted(chunk, styles["SmallCode"]))

    doc.build(story)


def main() -> None:
    build_notebook()
    build_pdf()
    print(f"Wrote notebook: {NOTEBOOK_PATH}")
    print(f"Wrote PDF: {PDF_PATH}")


if __name__ == "__main__":
    main()
