"""Checksums: streaming SHA-256/512 for dedupe and integrity verification."""
from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK = 1 << 20


def file_checksum(path: Path) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    sha512 = hashlib.sha512()
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK):
            sha256.update(chunk)
            sha512.update(chunk)
    return sha256.hexdigest(), sha512.hexdigest()


def verify_digest(path: Path, expected_sha256: str) -> bool:
    try:
        got, _ = file_checksum(path)
        return got == expected_sha256
    except OSError:
        return False
