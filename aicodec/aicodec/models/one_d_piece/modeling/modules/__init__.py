"""Vendored One-D-Piece ``modeling.modules`` package (inference-only subset).

The upstream ``__init__.py`` eagerly imports training-only modules
(``ema_model``, ``losses`` -> ``perceptual_loss``/``discriminator``) that are
NOT needed for the L-256 tokenizer inference path and are therefore not
vendored. Importing any submodule would otherwise execute those eager imports
and raise ImportError. Inference code uses fully-qualified imports
(``from modeling.modules.blocks import ...``), so no re-exports are required
here.
"""
