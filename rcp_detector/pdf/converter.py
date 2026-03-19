"""PDF to high-resolution PNG conversion using PyMuPDF."""

import logging
from pathlib import Path

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def pdf_to_pngs(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    colorspace: str = "rgb",
) -> list[Path]:
    """Render each page of *pdf_path* to a PNG at the given DPI.

    Returns a list of output PNG paths.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cs = fitz.csRGB if colorspace == "rgb" else fitz.csGRAY
    zoom = dpi / 72  # PyMuPDF default is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    output_paths: list[Path] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=matrix, colorspace=cs)
        stem = pdf_path.stem
        out_path = output_dir / f"{stem}_page{page_num + 1}.png"
        pix.save(str(out_path))
        output_paths.append(out_path)
        logger.info("Rendered %s page %d → %s (%dx%d)", pdf_path.name, page_num + 1, out_path.name, pix.width, pix.height)

    doc.close()
    return output_paths
