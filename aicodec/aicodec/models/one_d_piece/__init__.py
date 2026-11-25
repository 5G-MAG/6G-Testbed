"""One-D-Piece model package for aicodec.

Registers :class:`OneDPieceBridge` under the name ``"one_d_piece"``.

The package is NOT auto-imported by ``aicodec.models._load_builtin_models``
(that file is frozen to protect existing functionality); consumers register
it explicitly with a plain import:

    import aicodec.models.one_d_piece   # side effect: register_model(...)

Heavy deps (torch / omegaconf / torchvision) are imported inside
``OneDPieceBridge.setup()`` only, so this import stays cheap.
"""

from .. import register_model
from .bridge import OneDPieceBridge

register_model("one_d_piece", OneDPieceBridge)

__all__ = ["OneDPieceBridge"]
