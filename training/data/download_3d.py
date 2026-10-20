"""Fetch / unpack the 3D datasets into ~/neuroedge_data/raw/ and keep only what the evaluation needs.

    python -m training.data.download_3d ixi                       # unpack raw/ixi/IXI-T1.tar (downloaded separately)
    python -m training.data.download_3d synapse --id syn######## --dest brats_gli [--limit 300]
    python -m training.data.download_3d slim --dest brats_gli     # keep only *-t1c.nii.gz and *-seg.nii.gz

Synapse (BraTS): the user creates the account, accepts each dataset's terms on synapse.org and logs in once with
`synapse login` (personal access token, stored by synapseclient in ~/.synapseConfig). This script never asks for or
stores credentials. Archives are extracted one at a time and deleted after extraction, so disk use stays small.
Licences / terms are recorded in docs/DATA_SOURCES.md.
"""

from __future__ import annotations

import argparse
import hashlib
import tarfile
import zipfile
from pathlib import Path

from training.data.paths import RAW

KEEP_SUFFIXES = ("-t1c.nii.gz", "-seg.nii.gz")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_member(name: str) -> bool:
    p = Path(name)
    return not p.is_absolute() and ".." not in p.parts


def unpack_ixi(delete: bool = True) -> int:
    tar = RAW / "ixi" / "IXI-T1.tar"
    out = RAW / "ixi" / "IXI-T1"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar) as t:
        for m in t:
            if m.isfile() and m.name.endswith("-T1.nii.gz") and _safe_member(m.name):
                target = out / Path(m.name).name
                if not target.exists():
                    with t.extractfile(m) as src, target.open("wb") as dst:
                        dst.write(src.read())
                n += 1
    (RAW / "ixi" / "IXI-T1.sha256").write_text(f"{sha256(tar)}  IXI-T1.tar\n", encoding="utf-8")
    if delete:
        tar.unlink()
    return n


def slim(dest: Path) -> int:
    """Delete everything except post-contrast T1 and segmentation volumes (BraTS layout)."""
    removed = 0
    for f in list(dest.rglob("*.nii.gz")):
        if not f.name.endswith(KEEP_SUFFIXES):
            f.unlink()
            removed += 1
    return removed


def extract_keep(archive: Path, dest: Path, limit: int | None) -> int:
    """Extract only T1c + seg files from a BraTS zip/tar, at most `limit` cases, then delete the archive."""
    cases: set[str] = set()
    n = 0
    opener = zipfile.ZipFile if zipfile.is_zipfile(archive) else tarfile.open
    with opener(archive) as z:
        names = z.namelist() if isinstance(z, zipfile.ZipFile) else z.getnames()
        for name in sorted(names):
            if not (name.endswith(KEEP_SUFFIXES) and _safe_member(name)):
                continue
            case = Path(name).name.rsplit("-", 1)[0]
            if limit and case not in cases and len(cases) >= limit:
                continue
            cases.add(case)
            target = dest / case / Path(name).name
            target.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(z, zipfile.ZipFile):
                target.write_bytes(z.read(name))
            else:
                target.write_bytes(z.extractfile(name).read())
            n += 1
    archive.unlink()
    return n


def synapse(entity: str, dest: Path, limit: int | None) -> int:
    import synapseclient
    import synapseutils
    syn = synapseclient.Synapse()
    syn.login(silent=True)                              # uses the token saved by `synapse login`
    dest.mkdir(parents=True, exist_ok=True)
    stage = dest / "_download"
    files = synapseutils.syncFromSynapse(syn, entity, path=str(stage))
    total = 0
    for f in files:
        p = Path(f.path)
        if p.suffix in (".zip", ".tar", ".gz") and not p.name.endswith(".nii.gz"):
            total += extract_keep(p, dest, limit)
        elif p.name.endswith(KEEP_SUFFIXES):
            target = dest / p.parent.name / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            p.replace(target)
            total += 1
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ixi")
    s = sub.add_parser("synapse")
    s.add_argument("--id", required=True)
    s.add_argument("--dest", required=True)
    s.add_argument("--limit", type=int, default=None, help="keep at most N cases")
    sl = sub.add_parser("slim")
    sl.add_argument("--dest", required=True)
    a = ap.parse_args()
    if a.cmd == "ixi":
        print(f"IXI: {unpack_ixi()} T1 volumes in {RAW / 'ixi' / 'IXI-T1'}")
    elif a.cmd == "synapse":
        print(f"{synapse(a.id, RAW / a.dest, a.limit)} files kept in {RAW / a.dest}")
    else:
        print(f"removed {slim(RAW / a.dest)} non-T1c/seg files from {RAW / a.dest}")


if __name__ == "__main__":
    main()
