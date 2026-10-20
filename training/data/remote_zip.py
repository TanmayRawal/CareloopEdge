"""Read single files out of a large remote zip with HTTP range requests (no full download).

    python -m training.data.remote_zip list  <url>
    python -m training.data.remote_zip fetch <url> --dest DIR --match REGEX [--limit N]

Used for the NHNN pituitary release on figshare (one 47 GB zip, CC BY): only the T1c volumes and tumour masks are
needed, which is a small part of it. Supports zip64. Each fetched member is checked against its CRC-32 from the
archive index.
"""

from __future__ import annotations

import argparse
import re
import struct
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


def _resolve(url: str) -> str:
    """figshare download links redirect to a signed storage URL; resolve once."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.geturl()


_SOURCE: dict[str, str] = {}     # signed storage URL -> original download link (signed links expire quickly)


def _get(url: str, start: int, end: int, tries: int = 8) -> bytes:
    import http.client
    import time
    import urllib.error
    for attempt in range(tries):
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                if r.status != 206:
                    raise IOError("server does not support range requests")
                return r.read()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionError, OSError,
                http.client.IncompleteRead) as e:
            if attempt == tries - 1:
                raise
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403) and url in _SOURCE:
                fresh = _resolve(_SOURCE[url])          # the signed link expired: get a new one
                _SOURCE[fresh] = _SOURCE[url]
                _RENEWED[url] = fresh
                url = fresh
            time.sleep(2 * (attempt + 1))
    raise IOError("unreachable")


_RENEWED: dict[str, str] = {}


def current(url: str) -> str:
    while url in _RENEWED:
        url = _RENEWED[url]
    return url


def _size(url: str) -> int:
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


@dataclass
class Member:
    name: str
    method: int
    crc: int
    csize: int
    usize: int
    offset: int


def list_members(url: str) -> list[Member]:
    size = _size(url)
    tail = _get(url, max(0, size - 65_558), size - 1)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise IOError("end of central directory not found")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    if cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF:          # zip64
        j = tail.rfind(b"PK\x06\x07")
        (z64_off,) = struct.unpack("<Q", tail[j + 8:j + 16])
        z64 = _get(url, z64_off, z64_off + 55)
        cd_size, cd_off = struct.unpack("<QQ", z64[40:56])
    cd = _get(url, cd_off, cd_off + cd_size - 1)
    out, p = [], 0
    while p < len(cd) and cd[p:p + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[p + 10:p + 12])
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        (offset,) = struct.unpack("<I", cd[p + 42:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        q = 0
        while q + 4 <= len(extra):                              # zip64 extra field: real sizes / offset
            tag, ln = struct.unpack("<HH", extra[q:q + 4])
            if tag == 0x0001:
                vals = list(struct.unpack("<" + "Q" * (ln // 8), extra[q + 4:q + 4 + ln]))
                if usize == 0xFFFFFFFF:
                    usize = vals.pop(0)
                if csize == 0xFFFFFFFF:
                    csize = vals.pop(0)
                if offset == 0xFFFFFFFF:
                    offset = vals.pop(0)
            q += 4 + ln
        out.append(Member(name, method, crc, csize, usize, offset))
        p += 46 + nlen + elen + clen
    return out


def fetch(url: str, m: Member, dest: Path) -> Path:
    head = _get(current(url), m.offset, m.offset + 29)
    nlen, elen = struct.unpack("<HH", head[26:30])
    start = m.offset + 30 + nlen + elen
    chunk = 16 << 20                                  # 16 MB ranges: a dropped connection costs one chunk, not the file
    data = b"".join(_get(current(url), o, min(o + chunk, start + m.csize) - 1)
                    for o in range(start, start + m.csize, chunk)) if m.csize else b""
    if m.method == 8:
        data = zlib.decompress(data, -15)
    elif m.method != 0:
        raise IOError(f"unsupported compression method {m.method} for {m.name}")
    if zlib.crc32(data) & 0xFFFFFFFF != m.crc:
        raise IOError(f"CRC mismatch for {m.name}")
    target = dest / Path(m.name).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "fetch"])
    ap.add_argument("url")
    ap.add_argument("--dest", type=Path)
    ap.add_argument("--match", default=".")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    url = _resolve(a.url)
    _SOURCE[url] = a.url
    members = [m for m in list_members(url) if not m.name.endswith("/")]
    if a.cmd == "list":
        for m in members:
            print(f"{m.csize:>12} {m.usize:>12} {m.name}")
        print(f"{len(members)} files, {sum(m.csize for m in members) / 1e9:.1f} GB compressed")
        return
    rx = re.compile(a.match)
    sel = [m for m in members if rx.search(m.name)][: a.limit]
    print(f"fetching {len(sel)} files, {sum(m.csize for m in sel) / 1e9:.2f} GB", flush=True)
    for i, m in enumerate(sel, 1):
        if (a.dest / Path(m.name).name).exists():
            continue
        fetch(url, m, a.dest)
        if i % 10 == 0:
            print(f"  {i}/{len(sel)}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
