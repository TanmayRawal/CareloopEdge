"""Pi deployment files must be valid on Linux: LF line endings (CRLF breaks bash and systemd) and sane units."""

import configparser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = [ROOT / "edge" / "install_pi.sh", *sorted((ROOT / "edge" / "systemd").iterdir())]


def test_lf_line_endings():
    bad = [p.name for p in DEPLOY if b"\r" in p.read_bytes()]
    assert not bad, f"CRLF line endings in {bad}"


def test_units_parse_and_reference_known_paths():
    for p in DEPLOY[1:]:
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        cp.optionxform = str
        cp.read_string(p.read_text(encoding="utf-8").replace("\\n", " "))
        assert "Unit" in cp, p.name
        if p.suffix == ".service" and "User" in cp["Service"]:
            assert cp["Service"]["User"] == "neuroedge"
            assert cp["Service"].get("NoNewPrivileges") == "yes"
    web = (ROOT / "edge" / "systemd" / "neuroedge-web.service").read_text(encoding="utf-8")
    assert "--workers 1" in web          # in-memory case store requires a single process
