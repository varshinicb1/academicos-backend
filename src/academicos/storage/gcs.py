"""Google Cloud Storage object store implementation.

Uses the official client when available (optional extra `gcs`). Mirrors the
local layout under a configurable prefix (default `academicos`).
"""
from __future__ import annotations

from pathlib import Path

from .base import ObjectStore


class GcsStore(ObjectStore):
    def __init__(self, bucket: str, prefix: str = "academicos"):
        if not bucket:
            raise ValueError("GCS bucket not configured")
        from google.cloud import storage  # optional dependency

        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket)
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        k = key.strip("/")
        return f"{self.prefix}/{k}" if self.prefix else k

    def put(self, local_path: Path, key: str) -> None:
        blob = self.bucket.blob(self._key(key))
        blob.upload_from_filename(str(local_path))

    def get(self, key: str, local_path: Path) -> Path:
        blob = self.bucket.blob(self._key(key))
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        return local_path

    def exists(self, key: str) -> bool:
        return self.bucket.blob(self._key(key)).exists()

    def size(self, key: str) -> int:
        return self.bucket.blob(self._key(key)).size

    def keys(self, prefix: str = "") -> list[str]:
        out = []
        full = f"{self.prefix}/{prefix}" if prefix else self.prefix
        for blob in self.client.list_blobs(self.bucket, prefix=full):
            name = blob.name[len(self.prefix) + 1:]
            out.append(name)
        return out

    def delete(self, key: str) -> None:
        self.bucket.blob(self._key(key)).delete()
