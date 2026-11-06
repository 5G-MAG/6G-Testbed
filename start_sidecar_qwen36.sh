#!/usr/bin/env bash
# start_sidecar_qwen36.sh — bootstrap + launch the Qwen3.6 SGLang engine
# sidecar (aitestbed/server/sglang_engine_server_qwen36.py, port 8004).
#
# Standalone sibling of start_sidecar.sh (which launches the Qwen3-VL
# sidecar on 8002) — zero changes to that script, so all experiments
# can run/debug in parallel.
#
# Model: aicodec/aicodec/models/Qwen3.6-35B-A3B (qwen3_5_moe, hybrid
# GDN linear-attention MoE, BF16, ~70 GB weights — the BF16 default
# keeps the sidecar loadable on Ampere (cc 8.0) where the FP8
# snapshot cannot run; pass --model .../Qwen3.6-35B-A3B-FP8 on
# Hopper+). Needs a single 80GB-class GPU (or TP; forward extra args:
# --mem-fraction-static 0.7 etc. — see
# sglang_engine_server_qwen36.py --help).
#
# sglang patching: this script REUSES patch_host_sglang.py verbatim.
# The Qwen3.6 edge-mode PRECOMPUTED_EMBEDDING guard anchors in
# qwen3_vl.py::_get_visual_feature — the method
# Qwen3_5MoeForConditionalGeneration INHERITS (qwen3_5.py imports
# Qwen3VLForConditionalGeneration as its base) — so the same on-disk
# patch covers Qwen3.6. patch_host_sglang.py also patches the
# frame_ratio sampling branch (harmless when unused; max_frames alone
# is a native whitelist key). No NEW patch file exists for Qwen3.6 by
# design: a second anchored patch into the same qwen3_vl.py method
# would collide with the existing marker.
#
# Usage (from the 6G-Testbed root, inside the target conda env):
#   bash start_sidecar_qwen36.sh [--model PATH] [--host H] [--port P]
#        [--device D] [--server-cpus SPEC] [--mem-fraction-static F]
# Extra args are forwarded to the server. First launch JIT-compiles
# flashinfer kernels (a few minutes, cached in ~/.cache/sglang);
# subsequent launches are fast.
#
# --server-cpus (default auto-half = first half of the cores) caps the
# server arm's CPU budget (decode/preprocess is CPU-bound; scheduler
# subprocesses inherit the pin). Pair it with the edge worker's cpu
# pin (ENCPL_EDGE_CPUS / edge_cpus; the Qwen3.6 scenario defaults to
# auto-half-other) so BOTH arms are budgeted — on a single host keep
# the pools disjoint, e.g. server: --server-cpus 0-15.
# Verify via: curl http://<host>:8004/stats
#
# Idempotent: safe to re-run; symlinks are only created when missing.
# To use a system-wide CUDA toolkit instead of the conda one, export
# CUDA_HOME=<toolkit> before running (it must contain bin/nvcc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PY="$SCRIPT_DIR/aitestbed/server/sglang_engine_server_qwen36.py"
DEFAULT_PORT=8004

log() { echo "[start_sidecar_qwen36] $*"; }
die() { echo "[start_sidecar_qwen36] ERROR: $*" >&2; exit 1; }

[ -f "$SERVER_PY" ] || die "$SERVER_PY not found — place this script in the 6G-Testbed root, next to aitestbed/"

# --- 1. conda env must be active --------------------------------------------
[ -n "${CONDA_PREFIX:-}" ] || die "no conda env activated — run: conda activate <env>"

# --- 2. resolve CUDA_HOME ----------------------------------------------------
# Resolution order:
#   a) an externally exported, valid CUDA_HOME (real toolkit with bin/nvcc);
#   b) the conda env itself as the toolkit home (conda cuda-nvcc layout);
#   c) the pip-installed CUDA toolkit that torch cu13 wheels ship under
#      site-packages/nvidia/cu13/ (flat merged prefix: bin/nvcc, include/, lib/).
if [ -n "${CUDA_HOME:-}" ] && [ -x "$CUDA_HOME/bin/nvcc" ] && [ "$CUDA_HOME" != "$CONDA_PREFIX" ]; then
    log "using external CUDA_HOME=$CUDA_HOME"
elif [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    export CUDA_HOME="$CONDA_PREFIX"
else
    PIP_CUDA_HOME=""
    for _sp in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/cu13; do
        if [ -x "$_sp/bin/nvcc" ]; then
            PIP_CUDA_HOME="$_sp"
            break
        fi
    done
    if [ -n "$PIP_CUDA_HOME" ]; then
        export CUDA_HOME="$PIP_CUDA_HOME"
        log "using pip-installed CUDA toolkit: CUDA_HOME=$CUDA_HOME"
    else
        export CUDA_HOME="$CONDA_PREFIX"
    fi
fi
export PATH="$CUDA_HOME/bin:$PATH"

# --- 3. nvcc must be present -------------------------------------------------
if ! command -v nvcc >/dev/null 2>&1; then
    TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || true)"
    cat >&2 <<EOF
ERROR: nvcc not found — flashinfer JIT compilation will fail.
This machine only has the NVIDIA driver; the CUDA toolkit must live in
the conda env. Fix (network required; align versions with torch.version.cuda):

    conda install -y -n ${CONDA_DEFAULT_ENV:-<env>} -c nvidia \\
        cuda-nvcc=${TORCH_CUDA:-<ver>} cuda-cudart-dev=${TORCH_CUDA:-<ver>}

Alternative when the nvidia conda channel is unreachable (404 behind
mirrors): install via pip from the active env — this script auto-detects
the resulting site-packages/nvidia/cu13/ toolkit prefix:

    python -m pip install nvidia-cuda-nvcc==${TORCH_CUDA:-<ver>}.* \\
        nvidia-cuda-runtime==${TORCH_CUDA:-<ver>}.*

Then re-run this script.
EOF
    exit 1
fi

# --- 4. toolkit symlink fixes (for env-embedded toolkits: conda or pip cu13) --
# Applies whenever the resolved CUDA_HOME is a prefix that uses lib/ instead
# of lib64/ (conda envs and the pip cu13 layout both do).
if [ "$CUDA_HOME" != "$CONDA_PREFIX" ] || [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
    # nvcc/ninja search $CUDA_HOME/lib64; these layouts use lib/
    if [ -d "$CUDA_HOME/lib" ] && [ ! -e "$CUDA_HOME/lib64" ]; then
        ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
        log "created $CUDA_HOME/lib64 -> lib"
    fi
    # linkers need the dev name libcuda.so; the driver only ships libcuda.so.1
    if [ -d "$CUDA_HOME/lib64" ] && [ ! -e "$CUDA_HOME/lib64/libcuda.so" ]; then
        LIBCUDA="$(ldconfig -p 2>/dev/null | awk '$1 == "libcuda.so.1" {print $NF; exit}')"
        [ -n "$LIBCUDA" ] && [ -e "$LIBCUDA" ] \
            || die "cannot locate libcuda.so.1 via ldconfig — create $CUDA_HOME/lib64/libcuda.so manually"
        ln -sf "$LIBCUDA" "$CUDA_HOME/lib64/libcuda.so"
        log "created $CUDA_HOME/lib64/libcuda.so -> $LIBCUDA"
    fi
    # linkers need the dev name libcudart.so too; pip nvidia-cuda-runtime only
    # ships the versioned soname (libcudart.so.13) — create the dev symlink.
    # (conda toolkits ship libcudart.so via cuda-cudart-dev; branch self-skips.)
    if [ -d "$CUDA_HOME/lib64" ] && [ ! -e "$CUDA_HOME/lib64/libcudart.so" ]; then
        CUDART="$(ls "$CUDA_HOME/lib64"/libcudart.so.* 2>/dev/null | sort -V | tail -n 1)"
        if [ -n "$CUDART" ]; then
            ln -sfn "$CUDART" "$CUDA_HOME/lib64/libcudart.so"
            log "created $CUDA_HOME/lib64/libcudart.so -> $CUDART"
        fi
    fi
fi

# --- 5. port collision warning ------------------------------------------------
if ss -tln 2>/dev/null | grep -q ":$DEFAULT_PORT "; then
    log "WARNING: port $DEFAULT_PORT already listening — kill the old Qwen3.6 sidecar first if this is a stale one"
fi

# --- 6. patch the active env's sglang (edge-mode precomputed-embedding guard) --
# The scheduler child processes are spawned interpreters that re-import
# sglang from THIS env's site-packages; parent-process monkey-patches
# never reach them. The guard must live in the on-disk source.
# REUSES patch_host_sglang.py (the Qwen3-VL patcher): its on-disk patch
# anchors in qwen3_vl.py::_get_visual_feature, which the Qwen3.6
# classes (Qwen3_5MoeForConditionalGeneration) inherit — the same patch
# covers Qwen3.6. Idempotent (marker-guarded); exits 1 (aborting the
# launch) only if the installed sglang layout is not the expected
# pristine 0.5.20 one.
PATCHER_PY="$SCRIPT_DIR/aitestbed/server/patch_host_sglang.py"
if [ -f "$PATCHER_PY" ]; then
    log "patching sglang in the active env (qwen3_vl precomputed-embedding guard; covers Qwen3.6 via inheritance)..."
    python "$PATCHER_PY" || die "patch_host_sglang.py failed — sglang layout changed?"
else
    log "WARNING: $PATCHER_PY not found — skipping sglang patch (edge mode may crash)"
fi

# --- 7. launch ----------------------------------------------------------------
log "CUDA_HOME=$CUDA_HOME"
log "launching: python aitestbed/server/sglang_engine_server_qwen36.py $*"
exec python "$SERVER_PY" "$@"
