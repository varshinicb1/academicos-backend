"""Google Cloud Storage object store implementation.

Uses the official client when available (optional extra `gcs`). Mirrors the
local layout under a configurable prefix (default `academicos`).

Two surfaces: the file-based ObjectStore contract (CLI corpus sync) and a
bytes-level `put_bytes`/`get_bytes` pair with GCS object generations, used by
storage/blobs.py for scan media and the curriculum snapshot. The generation is
what lets a writer say "replace this object only if it is still the version I
loaded" -- see blobs.py for why the curriculum snapshot needs exactly that.
"""
from __future__ import annotations

from pathlib import Path

from .base import ObjectStore


class GcsStore(ObjectStore):
    def __init__(self, bucket: str, prefix: str = "academicos", client=None):
        if not bucket:
            raise ValueError("GCS bucket not configured")
        if client is None:
            from google.cloud import storage  # optional dependency

            client = storage.Client()
        # Injectable so tests can hand in an in-memory fake instead of
        # reaching a real project.
        self.client = client
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

    def put_bytes(self, key: str, data: bytes, content_type: str,
                  if_generation_match: int | None = None) -> int:
        """Write `data` and return the new object generation.

        `if_generation_match` is passed straight to GCS: 0 means "only if the
        object does not exist", N means "only if it is still generation N".
        A mismatch raises the client's PreconditionFailed (HTTP 412)."""
        blob = self.bucket.blob(self._key(key))
        kwargs = {"content_type": content_type}
        if if_generation_match is not None:
            kwargs["if_generation_match"] = if_generation_match
        blob.upload_from_string(data, **kwargs)
        return int(blob.generation or 0)

    def get_bytes(self, key: str) -> tuple[bytes, int] | None:
        """(data, generation), or None when the object does not exist.

        The read is pinned to the generation the metadata call returned, so
        the pair can never describe two different versions of the object."""
        blob = self.bucket.get_blob(self._key(key))
        if blob is None:
            return None
        generation = int(blob.generation or 0)
        data = blob.download_as_bytes(if_generation_match=generation or None)
        return data, generation

    def stat(self, key: str) -> tuple[int, str | None] | None:
        """(generation, md5 as GCS reports it -- base64 of the raw digest), or
        None when the object does not exist. Metadata only, no download."""
        blob = self.bucket.get_blob(self._key(key))
        if blob is None:
            return None
        return int(blob.generation or 0), getattr(blob, "md5_hash", None)

    def exists(self, key: str) -> bool:
        return self.bucket.blob(self._key(key)).exists()

    def size(self, key: str) -> int:
        if hasattr(self.bucket, "get_blob"):
            blob = self.bucket.get_blob(self._key(key))
            if blob is not None:
                return blob.size or 0
        blob = self.bucket.blob(self._key(key))
        return getattr(blob, "size", 0) or 0

    def keys(self, prefix: str = "") -> list[str]:
        out = []
        full = f"{self.prefix}/{prefix}" if prefix else self.prefix
        strip = len(self.prefix) + 1 if self.prefix else 0
        for blob in self.client.list_blobs(self.bucket, prefix=full):
            out.append(blob.name[strip:])
        return out

    def delete(self, key: str) -> None:
        self.bucket.blob(self._key(key)).delete()
