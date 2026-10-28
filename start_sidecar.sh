#!/usr/bin/env bash
# start_sidecar.sh — bootstrap + launch the SGLang engine sidecar
# (aitestbed/server/sglang_engine_server.py, Qwen3-VL encoder-placement test).
#
# Encapsulates the environment pitfalls hit on dgspl3310 so that a new
# machine or a fresh conda env works out of the box:
#   1. flashinfer JIT needs nvcc  -> the toolkit must be discoverable as one of:
#      conda env (cuda-nvcc), or the pip cu13 layout that torch cu13 wheels
#      install under site-packages/nvidia/cu13/ (auto-detected)
#   2. nvcc/ninja search $CUDA_HOME/lib64, env-embedded toolkits only have
#      lib/ -> symlink $CUDA_HOME/lib64 -> lib
#   3. linkers need the dev name libcuda.so; the driver only installs
#      libcuda.so.1 -> symlink $CUDA_HOME/lib64/libcuda.so -> <real driver
#      lib located via ldconfig>
#   4. the spawned scheduler subprocesses import sglang from this env's
#      site-packages, so the edge-mode PRECOMPUTED_EMBEDDING guard for
#      qwen3_vl must be patched into the on-disk source (auto, idempotent;
#      see aitestbed/server/patch_host_sglang.py)
# The deep_ep import crash is already handled inside the server itself
# (sys.modules stub at the top of sglang_engine_server.py); nothing to
# do about it here.
#
# Usage (from the 6G-Testbed root, inside the target conda env):
#   bash start_sidecar.sh [--model PATH] [--host H] [--port P] [--device D]
#        [--server-cpus SPEC]
# Extra args are forwarded to the server. First launch JIT-compiles
# flashinfer kernels (a few minutes, cached in ~/.cache/sglang);
# subsequent launches are fast.
#
# --server-cpus (default auto-half = first half of the cores) caps the
# server arm's CPU budget (decode/preprocess is CPU-bound; scheduler
# subprocesses inherit the pin). Pair it with the edge worker's cpu
# pin so BOTH arms are budgeted — on a single host keep the pools
# disjoint, e.g. server: --server-cpus 0-15.
# Verify via: curl http://<host>:8002/stats
#
# Idempotent: safe to re-run; symlinks are only created when missing.
# To use a system-wide CUDA toolkit instead of the conda one, export
# CUDA_HOME=<toolkit> before running (it must contain bin/nvcc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_PY="$SCRIPT_DIR/aitestbed/server/sglang_engine_server.py"
DEFAULT_PORT=8002

log() { echo "[start_sidecar] $*"; }
die() { echo "[start_sidecar] ERROR: $*" >&2; exit 1; }

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
    log "WARNING: port $DEFAULT_PORT already listening — kill the old sidecar first if this is a stale one"
fi

# --- 6. patch the active env's sglang (edge-mode precomputed-embedding guard) --
# The scheduler child processes are spawned interpreters that re-import
# sglang from THIS env's site-packages; parent-process monkey-patches
# never reach them. The guard must live in the on-disk source.
# Idempotent; exits 1 (aborting the launch) only if the installed
# sglang layout is not the expected pristine 0.5.20 one.
PATCHER_PY="$SCRIPT_DIR/aitestbed/server/patch_host_sglang.py"
if [ -f "$PATCHER_PY" ]; then
    log "patching sglang in the active env (qwen3_vl precomputed-embedding guard)..."
    python "$PATCHER_PY" || die "patch_host_sglang.py failed — sglang layout changed?"
else
    log "WARNING: $PATCHER_PY not found — skipping sglang patch (edge mode may crash)"
fi

# --- 7. launch ----------------------------------------------------------------
log "CUDA_HOME=$CUDA_HOME"
log "launching: python aitestbed/server/sglang_engine_server.py $*"
exec python "$SERVER_PY" "$@"
