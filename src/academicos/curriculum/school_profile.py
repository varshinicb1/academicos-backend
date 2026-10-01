"""The school's own name, address, affiliation line and logo (REQUIREMENTS
EX-5: papers and answer keys print "the school's name, logo"; audit D41).

Until now nothing stored a school's name. The paper header printed the
default branding template's name, so a school that had not opened the
Template Maker printed "Default CBSE Template" at the top of every paper,
and one that had saved a format printed that format's name ("Half-yearly
format") where the school's name goes. The logo was a file path on the
server, so a logo chosen on the web never reached a printed paper.

The profile is one row per school in the curriculum database (snapshotted
with the rest of the school's set-up). The logo is kept on the container's
disk and in the durable blob store ("school-media", GCS on the GCP release),
and read back from either -- the way homework photos are.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

SCHOOL_PROFILE_SCHEMA = """
-- EX-5 / D41: the school's own name and logo, printed on its papers.
CREATE TABLE IF NOT EXISTS school_profiles (
  school_id         TEXT PRIMARY KEY,
  name              TEXT NOT NULL,
  address           TEXT,
  affiliation       TEXT,
  logo_type         TEXT,
  logo_blob_key     TEXT,
  updated_by        TEXT,
  updated_at        TEXT NOT NULL
);
"""

# Names that are placeholders, never a school's: the default branding
# template's, and the Template Maker's when its "Template name" is left blank.
PLACEHOLDER_NAMES = frozenset({"Default CBSE Template", "School Template"})
LOGO_TYPES = {"image/png": "png", "image/jpeg": "jpg"}
MAX_LOGO_BYTES = 1024 * 1024
BLOB_PURPOSE = "school-media"


@dataclass
class SchoolProfile:
    school_id: str
    name: str
    address: Optional[str] = None
    affiliation: Optional[str] = None
    logo_type: Optional[str] = None
    logo_blob_key: Optional[str] = None
    updated_by: Optional[str] = None
    updated_at: str = ""

    @property
    def has_logo(self) -> bool:
        return self.logo_type is not None


def _clean(value: Optional[str], limit: int, what: str, *, required: bool = False) -> Optional[str]:
    cleaned = " ".join((value or "").split())
    if required and not cleaned:
        raise ValueError(f"the school's {what} cannot be empty")
    if len(cleaned) > limit:
        raise ValueError(f"the school's {what} is at most {limit} characters")
    return cleaned or None


class SchoolProfileMixin:
    """CurriculumStore's methods for the school profile. ValueError is the
    route's 422."""

    def get_school_profile(self, school_id: str) -> Optional[SchoolProfile]:
        r = self._fetchone("SELECT * FROM school_profiles WHERE school_id=?", (school_id,))
        return SchoolProfile(**r) if r else None

    def set_school_profile(self, school_id: str, *, name: str, address: Optional[str],
                           affiliation: Optional[str], updated_by: str) -> tuple[Optional[SchoolProfile], SchoolProfile]:
        """Returns (before, after). The logo is kept."""
        name = _clean(name, 120, "name", required=True)
        if name in PLACEHOLDER_NAMES:
            raise ValueError("that is the template's placeholder, not a school's name")
        address = _clean(address, 200, "address")
        affiliation = _clean(affiliation, 120, "affiliation line")
        now = datetime.now(timezone.utc).isoformat()
        with self._conn_lock:
            before = self.get_school_profile(school_id)
            self._exec(
                "INSERT INTO school_profiles (school_id, name, address, affiliation, updated_by, updated_at) "
                "VALUES (?,?,?,?,?,?) ON CONFLICT(school_id) DO UPDATE SET name=excluded.name, "
                "address=excluded.address, affiliation=excluded.affiliation, updated_by=excluded.updated_by, "
                "updated_at=excluded.updated_at",
                (school_id, name, address, affiliation, updated_by, now))
        self._commit()
        return before, self.get_school_profile(school_id)

    def set_school_logo(self, school_id: str, *, logo_type: Optional[str], blob_key: Optional[str],
                        updated_by: str) -> SchoolProfile:
        """Record (or, with logo_type None, clear) the logo. The profile must
        exist: a logo belongs to a named school."""
        with self._conn_lock:
            if self.get_school_profile(school_id) is None:
                raise KeyError(school_id)
            self._exec("UPDATE school_profiles SET logo_type=?, logo_blob_key=?, updated_by=?, updated_at=? "
                       "WHERE school_id=?",
                       (logo_type, blob_key, updated_by, datetime.now(timezone.utc).isoformat(), school_id))
        self._commit()
        return self.get_school_profile(school_id)


# ---------------------------------------------------------------- the logo file

def image_is_printable(source) -> bool:
    """Whether ReportLab can decode the whole image (bytes or a path). It
    reads an image lazily, only while it builds the PDF, so a file whose
    header is right but whose data is not failed every paper, answer key and
    report the school exported (E2E run on production's commit, 2026-10-01)."""
    import io

    from reportlab.lib.utils import ImageReader
    try:
        reader = ImageReader(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else str(source))
        width, height = reader.getSize()
        reader.getRGBData()
        return width > 0 and height > 0
    except Exception:  # noqa: BLE001 - any decoder error means "cannot print"
        return False


_PRINTABLE: dict[tuple[str, int, int], bool] = {}


def printable_logo(path: Optional[Path]) -> Optional[Path]:
    """`path` when it exists and prints, else None (logged once per version
    of the file): the paper prints without the logo rather than not at all."""
    if path is None or not path.exists():
        return None
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _PRINTABLE:
        _PRINTABLE[key] = image_is_printable(path)
        if not _PRINTABLE[key]:
            log.warning("school logo %s cannot be decoded; papers print without it", path.name)
    return path if _PRINTABLE[key] else None


def logo_path(data_root: Path, profile: SchoolProfile) -> Path:
    return Path(data_root) / BLOB_PURPOSE / f"logo_{profile.school_id}.{LOGO_TYPES[profile.logo_type]}"


def _blobs():
    from ..storage.blobs import durable_blob_store
    return durable_blob_store(BLOB_PURPOSE)


def save_logo(data_root: Path, profile: SchoolProfile, content_type: str, data: bytes) -> Optional[str]:
    """Write the logo to disk and to the durable store. Returns the blob key,
    or None when it is on this container's disk only (the store is off or
    refused; logged and counted in /health/storage)."""
    path = Path(data_root) / BLOB_PURPOSE / f"logo_{profile.school_id}.{LOGO_TYPES[content_type]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    for other in LOGO_TYPES.values():           # a PNG replacing a JPEG leaves no stale file
        path.with_suffix(f".{other}").unlink(missing_ok=True)
    path.write_bytes(data)
    blobs = _blobs()
    if not blobs.enabled:
        return None
    key = f"logos/{path.name}"
    try:
        blobs.upload(key, data, content_type)
        return key
    except Exception:  # noqa: BLE001 - kept on disk; the failure is counted, not hidden
        from ..storage.blobs import record_upload_failure
        record_upload_failure(BLOB_PURPOSE)
        log.warning("school logo for %s kept on container disk only", profile.school_id, exc_info=True)
        return None


def remove_logo(data_root: Path, school_id: str) -> None:
    for ext in LOGO_TYPES.values():
        (Path(data_root) / BLOB_PURPOSE / f"logo_{school_id}.{ext}").unlink(missing_ok=True)


def local_logo(data_root: Path, profile: Optional[SchoolProfile]) -> Optional[Path]:
    """The logo as a file on this container, restored from the durable store
    when a new container does not have it yet. None when the school has no
    logo, or it cannot be read back (logged; the paper prints without it)."""
    if profile is None or not profile.has_logo:
        return None
    path = logo_path(data_root, profile)
    if path.exists():
        return path
    if profile.logo_blob_key:
        try:
            data = _blobs().download(profile.logo_blob_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            return path
        except Exception:  # noqa: BLE001 - a paper still prints without its logo
            log.warning("school logo for %s is not in the blob store", profile.school_id, exc_info=True)
    return None


def printed_branding(template, profile: Optional[SchoolProfile], logo: Optional[Path], school_id: str):
    """The branding a paper or report prints: the school's own name and logo
    over the branding template (its colour, fonts and tagline stay).

    The profile's name wins. Without a profile, a branding template the
    school named itself still prints its name (schools that set their name
    through /school-templates keep it), but a placeholder -- the default
    template's "Default CBSE Template", the Template Maker's blank-name
    "School Template" -- prints as nothing rather than as a school's name.
    (A teacher's paper header that names the school travels on the paper
    itself and still prints first.) Returns a SchoolTemplate, or None when
    there is nothing to print but the neutral header."""
    from ..assessment.schemas import SchoolTemplate

    if template is None and profile is None:
        return None
    base = template or SchoolTemplate(id="profile", school_id=school_id, name="")
    if profile is not None:
        update: dict = {"name": profile.name}
    else:
        update = {"name": "" if base.name in PLACEHOLDER_NAMES else base.name}
    if logo is not None:
        update["logo_url"] = str(logo)
    elif base.logo_url and not Path(base.logo_url).is_file():
        # A logo "path" chosen on the web is not a file on this server: the
        # PDF would fail to draw it. Print without it instead.
        update["logo_url"] = ""
    return base.model_copy(update=update)


def branding_for_school(school_id: str, template=None):
    """printed_branding for a school, reading its profile and logo from the
    running app's curriculum store. The export routes call this; without a
    curriculum module (a CLI, a test with no app) it still drops the
    placeholder name."""
    from . import routes as cr
    if cr._store is None or cr._cfg is None:
        return printed_branding(template, None, None, school_id)
    profile = cr._store.get_school_profile(school_id)
    return printed_branding(template, profile, local_logo(cr._cfg.data_root, profile), school_id)
