# Source from bash before installing dependencies or running EPyMARL.
FASTAP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
export FASTAP_ROOT
export TMPDIR="$FASTAP_ROOT/.tmp/epymarl" TMP="$FASTAP_ROOT/.tmp/epymarl" TEMP="$FASTAP_ROOT/.tmp/epymarl"
export XDG_CACHE_HOME="$FASTAP_ROOT/.cache/epymarl" XDG_DATA_HOME="$FASTAP_ROOT/.cache/epymarl/data"
export XDG_CONFIG_HOME="$FASTAP_ROOT/.cache/epymarl/config" XDG_STATE_HOME="$FASTAP_ROOT/.cache/epymarl/state"
export PIP_CACHE_DIR="$XDG_CACHE_HOME/pip" PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
export UV_CACHE_DIR="$XDG_CACHE_HOME/uv" UV_PYTHON_INSTALL_DIR="$FASTAP_ROOT/envs/python"
export UV_PYTHON_BIN_DIR="$XDG_CACHE_HOME/bin" PYTHONNOUSERSITE=1 PYTHONPYCACHEPREFIX="$XDG_CACHE_HOME/pycache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib" TORCH_HOME="$XDG_CACHE_HOME/torch"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export TORCH_EXTENSIONS_DIR="$XDG_CACHE_HOME/torch-extensions" TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$UV_PYTHON_BIN_DIR"
