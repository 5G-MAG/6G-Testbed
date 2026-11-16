"""One-D-Piece bridge for aicodec (edge-side video tokenizer, realtime link).

One-D-Piece (turing-motors, arXiv:2501.10064) is a 1-D image tokenizer with
quality-controllable compression: a ViT encoder maps a 256x256 image to a
variable-length sequence of discrete VQ tokens (L-256: up to 256 tokens,
codebook 4096 x 12). Prefixes of the sequence reconstruct progressively
better images, so ``num_tokens`` is a rate/quality knob — the analog of
liquid's fixed 1024-token Chameleon VQ frames, but shorter and tunable.

Phase-1 scope (this bridge):
  - ``tokenize``   : PIL image -> list[int] (<= num_tokens uint16-safe ids)
  - ``detokenize`` : token ids -> reconstructed PIL image (verification only)
  - ``generate``   : NOT IMPLEMENTED — no LLM is attached yet; the realtime
    link ships tokens to the cloud where a Phase-2 connector + LLM will
    consume them (see the Phase-2 hooks below).

Phase-2 hooks (already usable for offline connector training):
  - ``tokens_to_latents`` : ids -> quantized continuous vectors [N, 12]
    (codebook lookup; this is the "continuous view" of the discrete tokens,
    so route A (id vocab) and route B (latent + MLP connector) both work
    from the SAME wire format — the edge never changes).
  - ``encode_latents``    : PIL -> z_quantized [num_tokens, 12]
  - ``get_codebook``      : export the 4096 x 12 codebook tensor.

Model code IS vendored (self-contained, see layout note below): ``setup()``
puts ``one_d_piece/`` (this package dir) on ``sys.path`` by default and
imports ``modeling.one_d_piece`` from there. No external checkout or env
vars are required for a normal run; ``ODP_REPO``/``ODP_CHECKPOINT``/
``ODP_CONFIG`` only need to be set to point at an external One-D-Piece
checkout instead.

Environment overrides (all optional; bundled defaults work out of the box):
  ODP_REPO        external One-D-Piece source repo, if not using the bundled
                   copy (default: this package's own directory)
  ODP_CHECKPOINT  L-256 tokenizer weights
                  (default: this package's ``checkpoints/pytorch_model.bin``)
  ODP_CONFIG      model yaml (default: ``odp_l256.yaml`` next to this file)
  ODP_NUM_TOKENS  tokens kept per frame, 1..256 (default 256)
  ODP_DEVICE      cuda / cpu (default: cuda if available)
"""

import logging
import os
import sys
from pathlib import Path
from typing import Iterator, List, Optional

from ..base import VLMBridge

logger = logging.getLogger(__name__)

# Self-contained layout (flattened directly into this package, 2026-11-09;
# previously nested one level deeper under a vendor/ subfolder):
#   one_d_piece/modeling/...                    vendored One-D-Piece model source
#   one_d_piece/checkpoints/pytorch_model.bin  L-256 weights
# Env vars ODP_REPO / ODP_CHECKPOINT still override (e.g. an external checkout).
_PKG_DIR = Path(__file__).resolve().parent
BUNDLED_ODP_REPO = _PKG_DIR
BUNDLED_ODP_CKPT = _PKG_DIR / "checkpoints" / "pytorch_model.bin"
DEFAULT_ODP_REPO = str(BUNDLED_ODP_REPO)
DEFAULT_NUM_TOKENS = 256
MAX_LATENT_TOKENS = 256  # L-256 checkpoint capacity


class OneDPieceBridge(VLMBridge):
    """Edge-side One-D-Piece tokenizer bridge (``enc``/``dec`` roles).

    ``enc_dec``:
      - "enc": encoder + quantizer on GPU (tokenize path)
      - "dec": adds the pixel decoder (detokenize path)
      - "llm": same as "dec" for now; generate() raises until Phase 2
    """

    model_name = "one_d_piece"

    #: tokens per frame (rate/quality knob); env ODP_NUM_TOKENS wins
    num_tokens: int = DEFAULT_NUM_TOKENS

    def __init__(self, enc_dec: str) -> None:
        self._model = None
        self._transform = None
        self._torch = None
        super().__init__(enc_dec)

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _resolve_device(self) -> str:
        dev = os.environ.get("ODP_DEVICE", "").strip()
        if dev:
            return dev
        return super()._resolve_device()

    def setup(self) -> None:
        import torch
        from omegaconf import OmegaConf

        self._torch = torch

        repo_env = os.environ.get("ODP_REPO", "").strip()
        repo = Path(repo_env).expanduser() if repo_env else BUNDLED_ODP_REPO
        if not (repo / "modeling" / "one_d_piece.py").exists():
            raise FileNotFoundError(
                f"One-D-Piece model source not found at {repo!r} "
                f"(expected modeling/one_d_piece.py; set env ODP_REPO to a "
                f"checkout, or restore the bundled one_d_piece/modeling/ tree)")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))

        ckpt = os.environ.get("ODP_CHECKPOINT", "").strip()
        if ckpt:
            ckpt_path = Path(ckpt)
        elif repo_env:
            ckpt_path = repo / "checkpoints" / "pytorch_model.bin"
        else:
            ckpt_path = BUNDLED_ODP_CKPT
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"One-D-Piece L-256 weights not found at {ckpt_path!r} "
                f"(set env ODP_CHECKPOINT)")

        cfg_path = os.environ.get("ODP_CONFIG", "").strip()
        cfg_path = Path(cfg_path) if cfg_path else Path(__file__).parent / "odp_l256.yaml"
        config = OmegaConf.load(str(cfg_path))
        config.model.vq_model.finetune_decoder = True
        config.model.vq_model.strict_length_assertion = False

        num_tokens = int(os.environ.get("ODP_NUM_TOKENS", "") or self.num_tokens)
        if not 1 <= num_tokens <= MAX_LATENT_TOKENS:
            raise ValueError(
                f"num_tokens must be in 1..{MAX_LATENT_TOKENS}, got {num_tokens}")
        self.num_tokens = num_tokens

        from modeling.one_d_piece import OneDPiece  # external repo, unmodified

        model = OneDPiece(config)
        try:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: {len(missing)} missing / "
                f"{len(unexpected)} unexpected keys (first missing: {missing[:3]})")

        model.eval()
        model.requires_grad_(False)
        model.to(self.device)
        self._model = model

        from torchvision import transforms

        self._transform = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])

        logger.info(
            "OneDPieceBridge ready: device=%s num_tokens=%d ckpt=%s",
            self.device, self.num_tokens, ckpt_path)

    def _require_model(self):
        if self._model is None:
            raise RuntimeError("OneDPieceBridge not set up")
        return self._model

    # ------------------------------------------------------------------
    # VLMBridge interface
    # ------------------------------------------------------------------
    def tokenize(self, image) -> List[int]:
        """PIL image -> ``num_tokens`` discrete VQ ids (0..4095)."""
        model = self._require_model()
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, result = model.encode(x)
        ids = result["min_encoding_indices"].flatten()[: self.num_tokens]
        return ids.to(torch.int64).cpu().tolist()

    def detokenize(self, tokens) -> object:
        """Token ids -> reconstructed PIL image (verification / dec role)."""
        model = self._require_model()
        torch = self._torch
        from PIL import Image
        t = torch.as_tensor(list(tokens), dtype=torch.long,
                            device=self.device).unsqueeze(0)  # [1, N]
        with torch.no_grad():
            rec = model.decode_tokens(t)[0]
        rec = (rec.clamp(0, 1) * 255.0).to(torch.uint8)
        rec = rec.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(rec)

    def generate(self, image_tokens, user_prompt: str) -> Iterator[str]:
        raise NotImplementedError(
            "one_d_piece bridge has no LLM attached yet (Phase 2: cloud-side "
            "connector + LLM will consume the received tokens)")

    def estimate_token_number(self, text: str) -> int:
        # no text tokenizer in Phase 1; rough 4-chars-per-token heuristic
        return max(1, len(text) // 4)

    # ------------------------------------------------------------------
    # Phase-2 hooks (connector training / latent-space fusion)
    # ------------------------------------------------------------------
    def tokens_to_latents(self, tokens):
        """Discrete ids [N] -> quantized continuous vectors [N, 12].

        Codebook lookup: the exact inverse mapping used by the pixel decoder,
        i.e. the continuous view of the transmitted tokens. A cloud-side
        connector (route B) can consume this without any wire-format change.
        """
        model = self._require_model()
        torch = self._torch
        t = torch.as_tensor(list(tokens), dtype=torch.long, device=self.device)
        with torch.no_grad():
            return model.quantize.get_codebook_entry(t)

    def encode_latents(self, image):
        """PIL image -> z_quantized [num_tokens, 12] (offline feature dump)."""
        model = self._require_model()
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            z_quantized, _ = model.encode(x)
        # [1, 12, 1, N] -> [N, 12]
        z = z_quantized.squeeze(0).squeeze(1)  # [12, N]
        return z.t()[: self.num_tokens].contiguous()

    def get_codebook(self):
        """Export the VQ codebook weight tensor [4096, 12]."""
        model = self._require_model()
        return model.quantize.embedding.weight.detach()
