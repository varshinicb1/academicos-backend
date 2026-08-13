"""Ingestion pipeline: discover -> validate -> register -> copy to object store -> record.

Result: every accepted source is in the ledger (registry) with a canonical
source_id (content-addressed via SHA-512 prefix) and a storage key.
"""
from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..models.enums import DocType
from ..storage.base import LocalStore, ObjectStore
from ..storage.registry import SourceRegistry
from .checksum import file_checksum
from .classify import classify_from_name, classify_from_text
from .discovery import Discovery, SourceFile
from .quality import validate_file

log = logging.getLogger(__name__)


class IngestResult:
    def __init__(self) -> None:
        self.accepted: list[SourceFile] = []
        self.rejected: list[tuple[SourceFile, list[str]]] = []
        self.duplicates: list[SourceFile] = []


class IngestionPipeline:
    def __init__(self, corpus_root: Path, registry: SourceRegistry, store: ObjectStore):
        self.corpus_root = corpus_root
        self.registry = registry
        self.store = store
        self.discovery = Discovery(corpus_root)

    def run(self, doc_type_filter: DocType | None = None, max_files: int | None = None,
            skip_validated: bool = True, classify_sample_pages: int = 1,
            min_pdf_bytes: int = 2048) -> IngestResult:
        files = self.discovery.walk(max_files=max_files)
        result = IngestResult()

        for sf in files:
            ok, problems = validate_file(sf, min_pdf_bytes=min_pdf_bytes)
            if not ok:
                result.rejected.append((sf, problems))
                log.warning("rejected %s: %s", sf.path.name, problems)
                continue

            sf.doc_type = classify_from_name(sf)
            if doc_type_filter and sf.doc_type is not doc_type_filter:
                continue

            if skip_validated and self.registry.get(sf.source_id):
                result.duplicates.append(sf)
                continue

            if sf.doc_type is DocType.UNKNOWN:
                # cheap first-page probe for text PDFs
                try:
                    import fitz  # pymupdf

                    with fitz.open(sf.path) as doc:
                        first = doc[0].get_text()[:2000] if doc.page_count else ""
                        sf.doc_type = classify_from_text(sf, first)
                except Exception:
                    pass

            key = self._store_key(sf)
            self.store.put(sf.path, key)
            self.registry.register(
                sf.source_id,
                file_key=key,
                sha256=sf.sha256,
                sha512=sf.sha512,
                size_bytes=sf.size_bytes,
                doc_type=sf.doc_type.value,
                title=sf.title,
                subject=sf.subject,
                grade=sf.grade,
                academic_year=sf.academic_year,
                series=sf.series,
                status="registered",
            )
            result.accepted.append(sf)
        return result

    def _store_key(self, sf: SourceFile) -> str:
        kind = sf.doc_type.value if sf.doc_type else "unknown"
        return f"{kind}/{safe_name(sf.source_id)}.{sf.extension}"


def safe_name(s: str) -> str:
    """Canonical ids contain ':' which is illegal in Windows filenames (ADS)."""
    return s.replace(":", "_").replace("/", "_").replace("\\", "_")
