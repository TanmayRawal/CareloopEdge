"""Fetch only selected members of a large zip stored on Synapse, without downloading the whole archive.

    python -m training.data.synapse_zip list  syn51514132
    python -m training.data.synapse_zip fetch syn51514132 --dest brats_gli --cases 300

Uses the login saved by `synapse config` (the user's own token; never handled here). Synapse hands out short-lived
pre-signed storage links; training.data.remote_zip renews them automatically through `renew`. For BraTS, only the
post-contrast T1 (`-t1c.nii.gz`) and segmentation (`-seg.nii.gz`) of N cases (random, seed 0) are kept.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from training.data import remote_zip
from training.data.paths import RAW

KEEP = re.compile(r"-(t1c|seg)\.nii\.gz$")


def presigned(syn, entity_id: str) -> str:
    ent = syn.get(entity_id, downloadFile=False)
    fh = ent._file_handle["id"]
    res = syn.restGET(f"/file/{fh}?redirect=false&fileAssociateType=FileEntity&fileAssociateId={entity_id[3:]}",
                      endpoint=syn.fileHandleEndpoint)
    return res if isinstance(res, str) else res["preSignedURL"] if isinstance(res, dict) else str(res)


def main() -> None:
    import synapseclient
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch"])
    ap.add_argument("entity")
    ap.add_argument("--dest")
    ap.add_argument("--cases", type=int, default=300)
    a = ap.parse_args()
    syn = synapseclient.Synapse(skip_checks=True)
    syn.login(silent=True)
    url = presigned(syn, a.entity)
    remote_zip._RENEW[url] = lambda: presigned(syn, a.entity)
    members = [m for m in remote_zip.list_members(url) if not m.name.endswith("/") and "__MACOSX" not in m.name]
    if a.cmd == "list":
        for m in members[:40]:
            print(f"{m.csize:>12} {m.name}")
        cases = sorted({Path(m.name).parent.name for m in members})
        print(f"{len(members)} files, {len(cases)} case folders, {sum(m.csize for m in members) / 1e9:.1f} GB")
        return
    keep = [m for m in members if KEEP.search(m.name)]
    # random (seeded) rather than the first N: case ids are roughly grouped by contributing hospital
    import random
    every = sorted({Path(m.name).parent.name for m in keep})
    cases = sorted(random.Random(0).sample(every, min(a.cases, len(every))))
    sel = [m for m in keep if Path(m.name).parent.name in set(cases)]
    dest = RAW / a.dest
    print(f"{len(cases)} cases, {len(sel)} files, {sum(m.csize for m in sel) / 1e9:.2f} GB -> {dest}", flush=True)
    for i, m in enumerate(sel, 1):
        case_dir = dest / Path(m.name).parent.name
        if (case_dir / Path(m.name).name).exists():
            continue
        remote_zip.fetch(url, m, case_dir)
        if i % 50 == 0:
            print(f"  {i}/{len(sel)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
