import logging
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from tqdm import tqdm

from rcp_detector.config import load_config, load_class_config

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("rcp_detector")


def _configure_run_log(log_path: Path | None) -> None:
    if log_path is None:
        return
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    abs_path = str(log_path.resolve())
    for handler in root.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, 'baseFilename', None) == abs_path:
            return
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(file_handler)
    logger.info("Writing detailed run log to %s", log_path)


def _quiet_ocr_loggers() -> None:
    for name in ("ppocr", "paddleocr"):
        logging.getLogger(name).setLevel(logging.WARNING)


@click.group()
@click.option("--config", "config_path", default=None, help="Path to custom config YAML")
@click.pass_context
def cli(ctx, config_path):
    """RCP Fixture Detector — detect ceiling fixtures from Reflected Ceiling Plan PDFs."""
    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)


# ── analyze ──────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", "-o", default="output", help="Output directory")
@click.option("--format", "out_format", type=click.Choice(["json", "csv", "both"]), default="json")
@click.option("--no-ocr", is_flag=True, help="Skip OCR tag reading")
@click.option("--no-viz", is_flag=True, help="Skip visualization output")
@click.pass_context
def analyze(ctx, source, output, out_format, no_ocr, no_viz):
    """Full pipeline: PDF → PNG → tiles → detect → reconstruct → output."""
    cfg = ctx.obj["cfg"]
    source = Path(source)
    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_cfg = load_class_config()
    class_names = class_cfg.get("names", {})

    # Collect PDFs/images to process
    if source.is_dir():
        pdfs = sorted(source.glob("*.pdf"))
        pngs = sorted(source.glob("*.png"))
        sources = pdfs + pngs
    elif source.suffix.lower() == ".pdf":
        sources = [source]
    else:
        sources = [source]

    all_results: dict[str, list[dict]] = {}

    for src in tqdm(sources, desc="Processing", unit="file"):
        try:
            result = _process_single(src, output_dir, cfg, class_names, no_ocr, no_viz)
            all_results.update(result)
        except Exception as e:
            if cfg["batch"]["skip_errors"]:
                logger.error("Failed to process %s: %s", src.name, e)
            else:
                raise

    # Write final output
    from rcp_detector.output.formatter import write_json, write_csv

    if out_format in ("json", "both"):
        write_json(all_results, output_dir / "results.json", class_names)
    if out_format in ("csv", "both"):
        write_csv(all_results, output_dir / "results.csv", class_names)

    # Print summary
    total = sum(len(dets) for dets in all_results.values())
    click.echo(f"\nProcessed {len(all_results)} image(s), {total} total fixture(s) detected.")

    for img_name, dets in all_results.items():
        counts: dict[str, int] = {}
        for d in dets:
            name = class_names.get(d["class_id"], f"class_{d['class_id']}")
            counts[name] = counts.get(name, 0) + 1
        click.echo(f"  {img_name}: {len(dets)} fixtures — {counts}")


def _process_single(
    source: Path,
    output_dir: Path,
    cfg: dict,
    class_names: dict,
    no_ocr: bool,
    no_viz: bool,
) -> dict[str, list[dict]]:
    """Process a single PDF or image through the full pipeline."""
    from rcp_detector.pdf.converter import pdf_to_pngs
    from rcp_detector.tiling.patcher import tile_image
    from rcp_detector.detection.detector import FixtureDetector
    from rcp_detector.tiling.reconstructor import reconstruct_detections

    work_dir = output_dir / source.stem
    work_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: PDF → PNG (skip if already an image)
    if source.suffix.lower() == ".pdf":
        png_dir = work_dir / "pages"
        image_paths = pdf_to_pngs(source, png_dir, dpi=cfg["pdf"]["dpi"], colorspace=cfg["pdf"]["colorspace"])
    else:
        image_paths = [source]

    all_results: dict[str, list[dict]] = {}
    tile_cfg = cfg["tiling"]
    det_cfg = cfg["detection"]

    for image_path in image_paths:
        # Step 2: Tile
        tiles_dir = work_dir / f"{image_path.stem}_tiles"
        meta = tile_image(
            image_path, tiles_dir,
            patch_width=tile_cfg["patch_width"],
            patch_height=tile_cfg["patch_height"],
            overlap_factor=tile_cfg["overlap_factor"],
        )

        if meta.num_patches == 0:
            logger.warning("No tiles generated for %s", image_path.name)
            all_results[image_path.stem] = []
            continue

        # Step 3: Detect
        detector = FixtureDetector(
            model_path=det_cfg["model_path"],
            confidence=det_cfg["confidence_threshold"],
            iou=det_cfg["iou_threshold"],
            device=det_cfg["device"],
            imgsz=det_cfg["imgsz"],
        )
        det_output = work_dir / f"{image_path.stem}_detections"
        detector.predict_tiles(
            tiles_dir / "images",
            det_output,
            batch_size=det_cfg["batch_size"],
        )

        # Step 4: Reconstruct
        recon_output = work_dir / f"{image_path.stem}_reconstructed"
        detections_by_image = reconstruct_detections(
            tiles_dir=tiles_dir / "images",
            labels_dir=det_output / "labels",
            output_dir=recon_output,
            meta_path=tiles_dir / "tiling_meta.json",
            patch_size=(tile_cfg["patch_height"], tile_cfg["patch_width"]),
            overlap_factor=tile_cfg["overlap_factor"],
            nms_threshold=cfg["nms"]["overlap_threshold"],
        )

        for img_name, dets in detections_by_image.items():
            # Step 5: OCR (optional)
            if not no_ocr and dets:
                try:
                    from rcp_detector.ocr.tag_reader import read_fixture_tags
                    ocr_cfg = cfg["ocr"]
                    dets = read_fixture_tags(
                        image_path, dets,
                        expand_ratio=ocr_cfg["expand_ratio"],
                        fixture_code_pattern=ocr_cfg["fixture_code_pattern"],
                        quantity_pattern=ocr_cfg["quantity_pattern"],
                        lang=ocr_cfg["lang"],
                    )
                except ImportError:
                    logger.warning("PaddleOCR not installed — skipping tag reading")

            # Step 6: Visualization (optional)
            if not no_viz and dets:
                from rcp_detector.output.visualizer import draw_detections
                viz_path = work_dir / f"{img_name}_viz.png"
                draw_detections(image_path, dets, viz_path, class_names, cfg["output"]["visualization_line_width"])

            all_results[img_name] = dets

    return all_results


# ── prepare ──────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", "-o", default="prepared", help="Output directory")
@click.pass_context
def prepare(ctx, source, output):
    """PDF conversion + tiling for annotation prep."""
    cfg = ctx.obj["cfg"]
    source = Path(source)
    output_dir = Path(output)

    from rcp_detector.pdf.converter import pdf_to_pngs
    from rcp_detector.tiling.patcher import tile_image

    # Collect PDFs
    if source.is_dir():
        pdfs = sorted(source.glob("*.pdf"))
    else:
        pdfs = [source]

    tile_cfg = cfg["tiling"]

    for pdf_path in tqdm(pdfs, desc="Preparing", unit="pdf"):
        pdf_out = output_dir / pdf_path.stem
        png_dir = pdf_out / "pages"

        image_paths = pdf_to_pngs(pdf_path, png_dir, dpi=cfg["pdf"]["dpi"])

        for img_path in image_paths:
            tiles_dir = pdf_out / f"{img_path.stem}_tiles"
            tile_image(
                img_path, tiles_dir,
                patch_width=tile_cfg["patch_width"],
                patch_height=tile_cfg["patch_height"],
                overlap_factor=tile_cfg["overlap_factor"],
            )

    click.echo(f"Prepared {len(pdfs)} PDF(s) → {output_dir}")


# ── train ────────────────────────────────────────────────────────────────────


@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True), help="YOLO dataset YAML")
@click.option("--model", default=None, help="Base model weights path")
@click.option("--epochs", default=None, type=int)
@click.option("--batch", default=None, type=int)
@click.pass_context
def train(ctx, data, model, epochs, batch):
    """Train a YOLOv8 model on RCP fixture data."""
    cfg = ctx.obj["cfg"]
    train_cfg = cfg["training"]

    from rcp_detector.training.trainer import train as do_train

    best = do_train(
        data_yaml=data,
        model_path=model or cfg["detection"]["model_path"],
        epochs=epochs or train_cfg["epochs"],
        batch_size=batch or train_cfg["batch_size"],
        imgsz=train_cfg["imgsz"],
        patience=train_cfg["patience"],
        device=cfg["detection"]["device"],
    )
    click.echo(f"Training complete. Best weights: {best}")


# ── stats ────────────────────────────────────────────────────────────────────


@cli.command()
@click.argument("labels_dir", type=click.Path(exists=True))
@click.option("--classes", "classes_file", default=None, help="Classes text file")
def stats(labels_dir, classes_file):
    """Print label distribution statistics for a dataset."""
    from rcp_detector.data.label_stats import count_labels, print_label_stats

    if classes_file is None:
        class_cfg = load_class_config()
        names = list(class_cfg["names"].values())
    else:
        with open(classes_file) as f:
            names = [line.strip() for line in f if line.strip()]

    counts = count_labels(labels_dir, class_names=names)
    print_label_stats(counts)


# ── ocr-count ────────────────────────────────────────────────────────────────


@cli.command("ocr-count")
@click.argument("source", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output markdown file path")
@click.option("--log-file", default=None, help="Detailed run log file path")
@click.option("--dpi", default=300, type=int, help="PDF render DPI (300-600)")
@click.option("--no-tiling", is_flag=True, help="Use full-page OCR instead of tile-based")
@click.option("--no-fans", is_flag=True, help="Skip ceiling fan detection")
@click.option("--dedup-dist", default=40.0, type=float, help="Center distance threshold for dedup (px)")
@click.pass_context
def ocr_count(ctx, source, output, log_file, dpi, no_tiling, no_fans, dedup_dist):
    """OCR-based fixture counting with spatial dedup and fan detection."""

    from rcp_detector.ocr.fixture_counter import (
        count_fixtures_from_pdf, count_fixtures_ocr, count_fixtures_tiled, format_results_markdown,
    )

    source = Path(source)
    all_results = []

    if source.is_dir():
        sources = sorted(list(source.glob("*.pdf")) + list(source.glob("*.png")))
    else:
        sources = [source]

    default_log = Path(log_file) if log_file else (
        Path(output).with_suffix(".log") if output else (source.with_suffix(".log") if source.is_file() else Path("ocr-count.log"))
    )
    _configure_run_log(default_log)
    _quiet_ocr_loggers()
    logger.info("OCR count inputs: %d source(s)", len(sources))

    for src in tqdm(sources, desc="OCR counting", unit="file"):
        if src.suffix.lower() == ".pdf":
            results = count_fixtures_from_pdf(
                src, dpi=dpi, lang="en",
                use_tiling=not no_tiling,
                detect_fans=not no_fans,
                dedup_dist=dedup_dist,
            )
            all_results.extend(results)
        elif src.suffix.lower() == ".png":
            if no_tiling:
                result = count_fixtures_ocr(src, lang="en")
            else:
                result = count_fixtures_tiled(
                    src, lang="en",
                    detect_fans=not no_fans,
                    dedup_dist=dedup_dist,
                )
            all_results.append(result)

    md = format_results_markdown(all_results)
    click.echo(md)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write("# RCP Fixture Count — OCR Results\n\n")
            f.write(f"**DPI:** {dpi} | **Tiling:** {'off' if no_tiling else 'on'} | ")
            f.write(f"**Fan detection:** {'off' if no_fans else 'on'}\n\n")
            f.write(md)
        click.echo(f"\nSaved to {output}")


if __name__ == "__main__":
    cli()
