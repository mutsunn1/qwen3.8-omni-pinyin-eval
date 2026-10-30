"""Probe iFlytek ISE (语音评测流式版) with one audio + reference pinyin.

Protocol: wss://ise-api.xfyun.cn/v2/open-ise with HMAC-SHA256 auth
(host/date/request-line), then:

1. send ``common + business(cmd=ssb) + data{status:0}`` with the test text;
2. stream raw 16 kHz PCM16 mono frames as base64:
   first ``cmd=auw, aus=1, status=1``; middle ``aus=2``; last ``aus=4, status=2``;
3. collect response ``data.data`` (base64) until ``status=2``, decode the XML.

The test text uses ISE's pinyin annotation format for ``read_syllable``::

    \\ufeff<customizer: interphonic>
    hao3

Usage::

    .venv/bin/python scripts/probe_xfyun_ise.py --audio data/audio/... --pinyin ai1
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import email.utils
import hashlib
import hmac
import json
import os
import urllib.parse
import wave
from pathlib import Path

try:
    import websockets
except ImportError:  # pragma: no cover - dependency hint
    websockets = None  # type: ignore[assignment]

HOST = "ise-api.xfyun.cn"
PATH = "/v2/open-ise"
CHUNK_MS = 40
SAMPLE_RATE = 16000


def auth_url(api_key: str, api_secret: str) -> str:
    date = email.utils.formatdate(usegmt=True)
    signature_origin = f"host: {HOST}\ndate: {date}\nGET {PATH} HTTP/1.1"
    signature = base64.b64encode(
        hmac.new(api_secret.encode(), signature_origin.encode(), hashlib.sha256).digest()
    ).decode()
    authorization_origin = (
        f'api_key="{api_key}", algorithm="hmac-sha256", '
        f'headers="host date request-line", signature="{signature}"'
    )
    authorization = base64.b64encode(authorization_origin.encode()).decode()
    query = urllib.parse.urlencode(
        {"authorization": authorization, "host": HOST, "date": date}
    )
    return f"wss://{HOST}{PATH}?{query}"


def read_pcm(path: str | Path) -> bytes:
    with wave.open(str(path), "rb") as fh:
        assert fh.getframerate() == SAMPLE_RATE, f"expected 16 kHz, got {fh.getframerate()}"
        assert fh.getnchannels() == 1, "expected mono"
        assert fh.getsampwidth() == 2, "expected PCM16"
        return fh.readframes(fh.getnframes())


def ise_text(pinyin: str) -> str:
    return "\ufeff<customizer: interphonic>\n" + pinyin + "\n"


async def run(audio: Path, pinyin: str, app_id: str, api_key: str, api_secret: str,
              *, text_override: str | None = None, category: str = "read_syllable",
              ent: str = "cn_vip", extra_ability: str = "syll_phone_err_msg;multi_dimension;pitch",
              ttp_skip: bool = True) -> dict:
    if websockets is None:
        raise SystemExit("websockets is not installed (uv pip install websockets)")
    pcm = read_pcm(audio)
    chunk_bytes = SAMPLE_RATE * 2 * CHUNK_MS // 1000
    chunks = [pcm[i:i + chunk_bytes] for i in range(0, len(pcm), chunk_bytes)] or [b""]

    business = {
        "sub": "ise",
        "ent": ent,
        "category": category,
        "cmd": "ssb",
        "text": text_override if text_override is not None else ise_text(pinyin),
        "tte": "utf-8",
        "ttp_skip": ttp_skip,
        "aue": "raw",
        "auf": "audio/L16;rate=16000",
        "rstcd": "utf8",
        "rst": "entirety",
        "ise_unite": "1",
        "extra_ability": extra_ability,
        "plev": "0",
        "group": "adult",
        "check_type": "common",
    }
    first = {"common": {"app_id": app_id}, "business": business, "data": {"status": 0}}
    responses: list[dict] = []
    async with websockets.connect(auth_url(api_key, api_secret), max_size=None) as ws:
        await ws.send(json.dumps(first, ensure_ascii=False))
        for index, chunk in enumerate(chunks):
            aus = 1 if index == 0 else (4 if index == len(chunks) - 1 else 2)
            status = 2 if index == len(chunks) - 1 else 1
            message = {
                "business": {"cmd": "auw", "aus": aus},
                "data": {
                    "status": status,
                    "data": base64.b64encode(chunk).decode(),
                },
            }
            await ws.send(json.dumps(message))
            await asyncio.sleep(0.01)
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            payload = json.loads(raw)
            responses.append(payload)
            if payload.get("code") not in (0, None):
                break
            if (payload.get("data") or {}).get("status") == 2:
                break
    result = responses[-1]
    xml_text = ""
    if result.get("data", {}).get("data"):
        xml_text = base64.b64decode(result["data"]["data"]).decode("utf-8", errors="replace")
    return {"responses": responses, "xml": xml_text}


def summarize(xml_text: str) -> None:
    import re

    def attr(name: str) -> list[str]:
        return re.findall(rf'{name}="([^"]*)"', xml_text)

    nodes = ["total_score", "tone_score", "phone_score", "fluency_score", "integrity_score"]
    print("scores:", {n: attr(n)[:3] for n in nodes})
    print("sylls:", re.findall(r'<syll[^>]*>', xml_text)[:4])
    print("phones:", re.findall(r'<phone[^>]*>', xml_text)[:6])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--pinyin", required=True, help="e.g. ai1 / zhang1 / lv4")
    parser.add_argument("--xml-out", default=None)
    parser.add_argument("--raw-json-out", default=None)
    parser.add_argument("--text-override", default=None, help="raw test text (for probing)")
    parser.add_argument("--category", default="read_syllable")
    parser.add_argument("--ent", default="cn_vip")
    parser.add_argument("--no-extra", action="store_true")
    parser.add_argument("--no-ttp-skip", action="store_true")
    args = parser.parse_args()
    app_id = os.environ.get("XFYUN_APPID", "")
    api_key = os.environ.get("XFYUN_APIKEY", "")
    api_secret = os.environ.get("XFYUN_APISECRET", "")
    if not (app_id and api_key and api_secret):
        raise SystemExit("set XFYUN_APPID / XFYUN_APIKEY / XFYUN_APISECRET")
    result = asyncio.run(run(
        Path(args.audio), args.pinyin, app_id, api_key, api_secret,
        text_override=args.text_override,
        category=args.category,
        ent=args.ent,
        extra_ability="" if args.no_extra else "syll_phone_err_msg;multi_dimension;pitch",
        ttp_skip=not args.no_ttp_skip,
    ))
    if args.xml_out:
        Path(args.xml_out).write_text(result["xml"], encoding="utf-8")
    if args.raw_json_out:
        Path(args.raw_json_out).write_text(
            json.dumps(result["responses"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("code:", result["responses"][-1].get("code"), "| sid:",
          result["responses"][-1].get("sid"))
    print("xml bytes:", len(result["xml"]))
    print(result["xml"][:600])
    if result["xml"]:
        summarize(result["xml"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
