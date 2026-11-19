"""Wire framing for the One-D-Piece image link over UDP.

Shared by the UDP clients (fragmented / whole-image) and the UDP server so
the three files can never drift apart on the byte layout.

Why UDP at all
--------------
The HTTP/TCP sibling (``odp_image_http_client.py`` /
``odp_image_http_server.py``) cannot express packet loss: TCP retransmits
every lost segment, so a netem ``loss`` profile only ever shows up as added
LATENCY and the token stream handed to the decoder is always bit-perfect.
Over UDP a dropped datagram is a genuinely missing slice of the 12-bit token
stream, which is exactly the degradation the offline token-drop study
(``run_mmbench.sh --keep-ratio/--drop-mode``) models analytically.

Datagram layout
---------------
    APP_HEADER (18 B) | NAME (name_len B) | TOKEN_HEADER (12 B) | CHUNK

APP_HEADER = ``!4sBBHIIH``
    magic        4s  b"ODPU"
    version      B   = 1
    nchunks      B   total datagrams for this image (1..255)
    name_len     H   length of the UTF-8 filename that follows
    index        I   dataset index of the image
    frame_bytes  I   total 12-bit-packed bytes of the WHOLE image (e.g. 384)
    chunk_seq    H   0-based datagram sequence within the image

TOKEN_HEADER is byte-for-byte the RTP link's header (``!IIHH`` =
picture_id, timestamp, pos, size), so the server can concatenate whatever
chunks survived and hand them straight to :func:`odp_recover_tokens`
(vendored below from ``aicodec.odp_codec``, see that function's docstring)
with no translation layer.

Two deliberate design choices
-----------------------------
* **The filename rides in EVERY datagram.** UDP has no control plane and no
  ordering guarantee, so a "first datagram carries the metadata" scheme
  loses the entire image whenever that one datagram drops. Repeating ~30 B
  of name per datagram is cheap and makes every datagram self-describing.
* **Chunks split on EVEN token counts only.** The 12-bit packing stores two
  tokens per three bytes; an odd split would straddle a byte boundary and
  corrupt the token pair that spans the seam.

Control datagram (session end)
------------------------------
    ``!4sBB`` = magic b"ODPC", version, kind   (kind 1 = SESSION_COMPLETE)

Sent a few times by the client because it, too, can be lost; the server
treats it idempotently.
"""

import json
import random
import struct
from pathlib import Path
from typing import Optional

#: image file extensions recognized by discover_images()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: data datagram magic
ODP_UDP_MAGIC = b"ODPU"
#: control datagram magic
ODP_CTL_MAGIC = b"ODPC"
ODP_UDP_VERSION = 1

#: control kinds
CTL_SESSION_COMPLETE = 1

APP_HEADER_FMT = "!4sBBHIIH"
APP_HEADER_LENGTH = struct.calcsize(APP_HEADER_FMT)       # 18
CTL_HEADER_FMT = "!4sBB"
CTL_HEADER_LENGTH = struct.calcsize(CTL_HEADER_FMT)       # 6

#: mirrors the RTP link's picture_id/ts/pos/size header (picture_id, ts,
#: pos, size); see ``odp_recover_tokens`` below for why this module keeps
#: its own copy instead of importing ``aicodec.odp_codec``.
TOKEN_HEADER_FMT = "!IIHH"
TOKEN_HEADER_LENGTH = struct.calcsize(TOKEN_HEADER_FMT)   # 12

#: 12-bit ids -> 2 tokens per 3 bytes
ODP_TOKEN_BITS = 12


def packed_size(num_tokens: int) -> int:
    """Byte length of ``num_tokens`` 12-bit-packed token ids."""
    return (num_tokens * ODP_TOKEN_BITS + 7) // 8


def pack_tokens_12bit(tokens) -> bytes:
    """Pack token ids (each < 4096) into a 12-bit big-endian bitstream.

    Layout: two tokens per 3 bytes —
      byte0 = t0[11:4], byte1 = t0[3:0]<<4 | t1[11:8], byte2 = t1[7:0].
    An odd trailing token occupies 2 bytes with its low nibble zero-padded,
    so the count is exactly recoverable from the byte length.

    Vendored from ``aicodec.odp_codec`` (the aiortc tokenIdODP codec): that
    module unconditionally imports aiortc at load time, so the UDP
    client/server import this copy instead and never pull aiortc in just to
    pack/unpack a 12-bit token stream.
    """
    out = bytearray()
    it = iter(tokens)
    for t0 in it:
        t1 = next(it, None)
        if t1 is None:
            out += bytes(((t0 >> 4) & 0xFF, (t0 & 0xF) << 4))
        else:
            out += bytes(((t0 >> 4) & 0xFF,
                          ((t0 & 0xF) << 4) | ((t1 >> 8) & 0xF),
                          t1 & 0xFF))
    return bytes(out)


def unpack_tokens_12bit(data: bytes) -> tuple:
    """Inverse of :func:`pack_tokens_12bit`; count derived from byte length.

    len%3==0 -> 2 tokens per 3 bytes; len%3==2 -> one extra 2-byte token.
    A trailing partial byte (loss) is dropped.
    """
    n_pairs = len(data) // 3
    tokens = []
    for i in range(n_pairs):
        b0, b1, b2 = data[3 * i], data[3 * i + 1], data[3 * i + 2]
        tokens.append((b0 << 4) | (b1 >> 4))
        tokens.append(((b1 & 0xF) << 8) | b2)
    rem = len(data) - 3 * n_pairs
    if rem == 2:
        tokens.append((data[-2] << 4) | (data[-1] >> 4))
    return tuple(tokens)


def odp_recover_tokens(frame_data: bytes):
    """Reassemble a variable-length 12-bit token frame from concatenated
    ``TOKEN_HEADER (12B) + payload`` chunks (one per surviving datagram).

    Chunks may be missing (UDP loss); gaps leave zeros. The recovered
    length is the true frame size (max ``pos + size`` across chunks), not a
    constant. Returns ``(picture_id, timestamp, tokens_tuple)``.

    Vendored from ``aicodec.odp_codec.odp_recover_tokens`` for the same
    aiortc-free-import reason as :func:`pack_tokens_12bit` above.
    """
    picture_id = -1
    timestamp = 0
    chunks = []
    max_end = 0
    pointer = 0
    data_len = len(frame_data)
    while pointer + TOKEN_HEADER_LENGTH <= data_len:
        pid, ts, pos, size = struct.unpack_from(
            TOKEN_HEADER_FMT, frame_data, pointer)
        pointer += TOKEN_HEADER_LENGTH
        actual = min(size, data_len - pointer)
        chunks.append((pos, frame_data[pointer:pointer + actual]))
        pointer += actual
        picture_id = pid
        timestamp = ts
        max_end = max(max_end, pos + actual)

    filled = bytearray(max_end)
    for pos, chunk in chunks:
        end = min(pos + len(chunk), max_end)
        filled[pos:end] = chunk[:end - pos]

    tokens = unpack_tokens_12bit(bytes(filled)) if max_end else ()
    return picture_id, timestamp, tokens


def chunk_bytes_for_tokens(tokens_per_datagram: int) -> int:
    """Byte length of ``tokens_per_datagram`` 12-bit-packed ids.

    Rejects odd counts: two tokens share three bytes, so an odd split would
    cut a token pair in half and corrupt it on recovery.
    """
    if tokens_per_datagram <= 0:
        raise ValueError("tokens_per_datagram must be positive")
    if tokens_per_datagram % 2:
        raise ValueError(
            f"tokens_per_datagram must be EVEN (got {tokens_per_datagram}): "
            "12-bit packing stores 2 tokens per 3 bytes, so an odd split "
            "straddles a byte boundary and corrupts the straddling pair")
    return tokens_per_datagram * ODP_TOKEN_BITS // 8


def plan_chunks(frame_len: int, chunk_size: int):
    """``[(pos, size)]`` slices covering ``frame_len`` bytes."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out = []
    pos = 0
    while pos < frame_len:
        out.append((pos, min(chunk_size, frame_len - pos)))
        pos += chunk_size
    return out


def apply_payload_token_drop(tokens: list, drop_mode: str, drop_pct: float,
                              seed: Optional[int] = None) -> list:
    """Simulate payload-level synthetic token loss BEFORE 12-bit packing.

    Mirrors the offline token-drop study's ``tail``/``random`` strategies
    (see ``one_d_piece_folder_recon.py``), but applied on the LIVE UDP wire
    path so this degradation composes with real netem packet loss instead of
    only existing as an analytical, link-free model.

    drop_mode:
        "none"   -- no-op, returns ``tokens`` unchanged (sanity baseline).
        "tail"   -- keep the first ``(100 - drop_pct)%`` tokens (most
                    important, per One-D-Piece's ordered/coarse-to-fine token
                    stream) and truncate the rest. Sequence gets SHORTER.
        "random" -- zero out ``drop_pct%`` of tokens at uniformly random
                    positions (control experiment that ignores token
                    importance ordering). Sequence LENGTH is unchanged;
                    dropped slots are overwritten with token id 0 so the
                    downstream bridge still sees a well-formed fixed-length
                    sequence.

    drop_pct: 0..100.
    seed: optional RNG seed for "random" mode reproducibility (e.g. derive
        from the image index so repeated runs / pct sweeps are comparable).
    """
    if drop_mode in (None, "none") or drop_pct <= 0:
        return tokens
    n = len(tokens)
    if drop_mode == "tail":
        keep = max(0, round(n * (100.0 - drop_pct) / 100.0))
        return tokens[:keep]
    if drop_mode == "random":
        k = round(n * drop_pct / 100.0)
        if k <= 0:
            return tokens
        rng = random.Random(seed)
        drop_idx = set(rng.sample(range(n), min(k, n)))
        return [0 if i in drop_idx else t for i, t in enumerate(tokens)]
    raise ValueError(f"unknown payload_drop_mode: {drop_mode!r}")


def build_datagram(*, index: int, name: str, frame_bytes: int, nchunks: int,
                   chunk_seq: int, picture_id: int, timestamp: int,
                   pos: int, chunk: bytes) -> bytes:
    """Assemble one data datagram (see module docstring for the layout)."""
    if not 1 <= nchunks <= 255:
        raise ValueError(f"nchunks must fit one byte, got {nchunks}")
    raw_name = name.encode("utf-8")
    if len(raw_name) > 0xFFFF:
        raise ValueError("filename too long for the 16-bit name_len field")
    app = struct.pack(APP_HEADER_FMT, ODP_UDP_MAGIC, ODP_UDP_VERSION,
                      nchunks, len(raw_name), index, frame_bytes, chunk_seq)
    tok = struct.pack(TOKEN_HEADER_FMT, picture_id & 0xFFFFFFFF,
                      timestamp & 0xFFFFFFFF, pos, len(chunk))
    return app + raw_name + tok + chunk


def parse_datagram(data: bytes) -> dict:
    """Decode one data datagram. 数据

    Raises ``ValueError`` on anything malformed so the caller can count it
    as a corrupt datagram instead of crashing the receive loop.
    """
    if len(data) < APP_HEADER_LENGTH:
        raise ValueError(f"short datagram: {len(data)} B")
    magic, version, nchunks, name_len, index, frame_bytes, chunk_seq = \
        struct.unpack(APP_HEADER_FMT, data[:APP_HEADER_LENGTH])
    if magic != ODP_UDP_MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != ODP_UDP_VERSION:
        raise ValueError(f"unsupported version {version}")

    off = APP_HEADER_LENGTH
    end_name = off + name_len
    if len(data) < end_name + TOKEN_HEADER_LENGTH:
        raise ValueError("datagram truncated inside name/token header")
    name = data[off:end_name].decode("utf-8", errors="replace")

    tok = data[end_name:end_name + TOKEN_HEADER_LENGTH]
    picture_id, timestamp, pos, size = struct.unpack(TOKEN_HEADER_FMT, tok)
    chunk = data[end_name + TOKEN_HEADER_LENGTH:]
    if len(chunk) != size:
        raise ValueError(
            f"chunk length {len(chunk)} != declared size {size}")

    return {"nchunks": nchunks, "index": index, "frame_bytes": frame_bytes,
            "chunk_seq": chunk_seq, "name": name, "picture_id": picture_id,
            "timestamp": timestamp, "pos": pos, "size": size,
            "chunk": chunk, "token_header": tok}


def build_control(kind: int = CTL_SESSION_COMPLETE) -> bytes:
    """Assemble a control datagram."""
    return struct.pack(CTL_HEADER_FMT, ODP_CTL_MAGIC, ODP_UDP_VERSION, kind)


def parse_control(data: bytes):
    """Return the control ``kind``, or ``None`` if this is not a control
    datagram (so the receive loop can fall through to :func:`parse_datagram`).
    """
    if len(data) != CTL_HEADER_LENGTH:
        return None
    magic, version, kind = struct.unpack(CTL_HEADER_FMT, data)
    if magic != ODP_CTL_MAGIC or version != ODP_UDP_VERSION:
        return None
    return kind


def max_datagram_bytes(name: str, chunk_size: int) -> int:
    """Worst-case on-wire size, for MTU sanity checks."""
    return (APP_HEADER_LENGTH + len(name.encode("utf-8"))
            + TOKEN_HEADER_LENGTH + chunk_size)


def discover_images(image_dir: str, manifest_path: Optional[str] = None,
                     max_images: Optional[int] = None) -> list:
    """Ordered list of (index, name, source_abs_path).

    Shared dataset-discovery helper for both UDP clients (whole/chunked) and
    the orchestrator scenario layer, so all three agree on image order.

    Order of preference:
      1. explicit ``manifest_path`` jsonl;
      2. ``<image_dir>/manifest.jsonl`` (the dataset manifest, e.g.
         MMBench_DEV_EN ships index/image/question/answer per line) --
         this is the DATASET order, not filename-lexicographic order;
      3. filename-sorted directory scan (fallback).
    ``max_images``: cap; 0/None = no cap (full dataset).
    """
    base = Path(image_dir)
    manifest = Path(manifest_path) if manifest_path else None
    if manifest is None:
        auto = base / "manifest.jsonl"
        if auto.exists():
            manifest = auto
    if manifest is not None and manifest.exists():
        out = []
        with open(manifest, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rel = rec.get("image") or rec.get("path") or rec.get("name")
                if not rel:
                    continue
                p = Path(rel) if Path(rel).is_absolute() else base / rel
                if not p.exists():
                    continue
                try:
                    idx = int(rec.get("index", i))
                except (TypeError, ValueError):
                    idx = i
                out.append((idx, p.name, str(p)))
                if max_images and len(out) >= max_images:
                    break
        if out:
            return out
    files = sorted(
        (p for p in base.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
        key=lambda p: p.name)
    out = [(i, p.name, str(p)) for i, p in enumerate(files)]
    if max_images:
        out = out[:max_images]
    return out
