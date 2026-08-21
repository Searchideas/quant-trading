"""Windows CA bundle for yfinance / curl TLS (antivirus HTTPS inspection)."""

from __future__ import annotations

import os
import ssl
from pathlib import Path

from config import ROOT


def windows_ca_bundle() -> str:
    bundle = ROOT / "data" / "windows-ca-bundle.pem"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    pems: list[str] = []
    for store in ("CA", "ROOT"):
        try:
            certs = ssl.enum_certificates(store)
        except OSError:
            continue
        for item in certs:
            try:
                pems.append(ssl.DER_cert_to_PEM_cert(item[0]))
            except (TypeError, ValueError):
                continue
    if not pems:
        raise RuntimeError("Could not export Windows certificates")
    bundle.write_text("".join(pems), encoding="ascii")
    return str(bundle)


def configure_ssl() -> str:
    """Export Windows CAs and point curl/requests env vars at the bundle."""
    ca = windows_ca_bundle()
    os.environ["CURL_CA_BUNDLE"] = ca
    os.environ["SSL_CERT_FILE"] = ca
    os.environ["REQUESTS_CA_BUNDLE"] = ca
    return ca
