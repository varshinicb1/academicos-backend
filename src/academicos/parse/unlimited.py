"""Unlimited-OCR VLM parser (baidu/Unlimited-OCR lineage).

Architecture reference from https://github.com/baidu/Unlimited-OCR (DeepSeek-OCR
family, ~3B params). This provider shells out to a local transformers-based
service script so the corpus pipeline never hard-depends on the VLM runtime.

Two integration modes:
  1. `mode="service"`: POST page images to a running unlimited-ocr HTTP service
     (started via scripts/run_unlimited_ocr_service.py or the repo's app.py).
  2. `mode="cli"`: invoke the repo's cli/checkpoint_generate.py with torchrun /
     python directly (batch of page images).

The provider expects the service to return JSON:
  {"markdown": str, "json": {...}, "confidence": float, "language": str}
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..models.document import Page, ParseResult
from ..models.enums import PageKind
from .base import ParserProvider

DEFAULT_URL = "http://127.0.0.1:8910"


class UnlimitedOcrParser(ParserProvider):
    name = "unlimited_ocr"

    def __init__(self, mode: str = "service", url: str = DEFAULT_URL,
                 binary: str | None = None, device: str = "cuda", page_dpi: int = 200):
        self.mode = mode
        self.url = url
        self.binary = binary or shutil.which("python")
        self.device = device
        self.page_dpi = page_dpi

    def parse(self, document_path: Path, document_id: str, max_pages: int = 500) -> ParseResult:
        import fitz

        res = ParseResult(document_id=document_id, method=self.name)
        pages: list[Page] = []
        with fitz.open(document_path) as doc:
            for i in range(min(doc.page_count, max_pages)):
                pix = doc[i].get_pixmap(dpi=self.page_dpi)
                png = pix.tobytes("png")
                if self.mode == "service":
                    out = self._call_service(png)
                else:
                    out = self._call_cli(png)
                if out is None:
                    res.warnings.append(f"page {i + 1}: vlm unavailable")
                    pages.append(Page(page_no=i + 1, kind=PageKind.SCAN, text="", quality=0.0))
                    continue
                md = out.get("markdown", "")
                pages.append(Page(page_no=i + 1, kind=PageKind.MIXED, text=md, quality=1.0))
            res.pages = pages
            res.markdown = "\n\n".join(p.text for p in pages)
            res.language = "hi" if any(True for _ in []) else None
        res.confidence = 0.9 if res.pages and any(p.quality > 0 for p in res.pages) else 0.0
        return res

    def _call_service(self, png: bytes) -> dict | None:
        try:
            import requests
            r = requests.post(f"{self.url}/parse", files={"image": ("page.png", png, "image/png")}, timeout=300)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def _call_cli(self, png: bytes) -> dict | None:
        """Invoke unlimited-ocr checkpoint_generate.py on a temp png."""
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            tf.write(png)
            tmp = tf.name
        try:
            script = shutil.which("checkpoint_generate.py") or str(
                Path(__file__).resolve().parents[3] / "vendor" / "Unlimited-OCR" / "cli" / "checkpoint_generate.py"
            )
            if not Path(script).exists():
                return None
            proc = subprocess.run(
                [self.binary, script, tmp],
                capture_output=True, text=True, timeout=600,
                env={"DEVICE": self.device, **__import__("os").environ},
            )
            if proc.returncode != 0:
                return None
            return json.loads(proc.stdout)
        except Exception:
            return None
        finally:
            Path(tmp).unlink(missing_ok=True)
