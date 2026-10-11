# Keep every tool, interpreter and cache inside the project (nothing installed globally).
# Usage: source scripts/local_env.sh
DB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$DB_ROOT/envs/_tools/bin:$PATH"
export XDG_CACHE_HOME="$DB_ROOT/.cache"
export UV_CACHE_DIR="$DB_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$DB_ROOT/.cache/python"
export UV_PYTHON_PREFERENCE=only-managed
export PIP_CACHE_DIR="$DB_ROOT/.cache/pip"
export HF_HOME="$DB_ROOT/.cache/huggingface"
export TORCH_HOME="$DB_ROOT/.cache/torch"
export YOLO_CONFIG_DIR="$DB_ROOT/.cache/ultralytics"
export MPLCONFIGDIR="$DB_ROOT/.cache/matplotlib"
export KAGGLEHUB_CACHE="$DB_ROOT/.cache/kagglehub"
export PYTHONIOENCODING=utf-8
export UV_HTTP_TIMEOUT=600
export UV_HTTP_RETRIES=10
