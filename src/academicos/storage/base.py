"""Storage abstraction: a file/object store contract with local + GCS impls."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class ObjectStore(ABC):
    @abstractmethod
    def put(self, local_path: Path, key: str) -> None: ...

    @abstractmethod
    def get(self, key: str, local_path: Path) -> Path: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def size(self, key: str) -> int: ...

    @abstractmethod
    def keys(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...


class LocalStore(ObjectStore):
    """Plain filesystem implementation. The root is the documents root."""
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        return (self.root / key).resolve()

    def put(self, local_path: Path, key: str) -> None:
        dst = self._resolve(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(Path(local_path).read_bytes())

    def get(self, key: str, local_path: Path) -> Path:
        src = self._resolve(key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(src.read_bytes())
        return local_path

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def size(self, key: str) -> int:
        return self._resolve(key).stat().st_size

    def keys(self, prefix: str = "") -> list[str]:
        if not self.root.exists():
            return []
        base = self.root / prefix
        if not base.exists():
            return []
        return [str(p.relative_to(self.root)).replace("\\", "/") for p in base.rglob("*") if p.is_file()]

    def delete(self, key: str) -> None:
        p = self._resolve(key)
        if p.exists():
            p.unlink()
