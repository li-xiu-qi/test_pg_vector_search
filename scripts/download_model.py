#!/usr/bin/env python3
"""增强版模型下载脚本（中文提示）。

功能：
- 接受可选的模型名参数（默认：BAAI/bge-m3）
- 检查 `modelscope` 是否可用或可导入，必要时自动安装
- 使用 ModelScope 的 Python API（snapshot_download）下载模型，避免 `python -m modelscope` 的入口限制

用法：
    python scripts/download_model.py [BAAI/bge-m3]
"""
from __future__ import annotations

import sys
import os
import subprocess
import warnings

DEFAULT_MODEL = "BAAI/bge-m3"


def run(cmd: list[str]) -> int:
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        return 127


def ensure_modelscope_installed() -> bool:
    """检查 modelscope 是否可导入；如果不可用，尝试用 pip 安装，成功返回 True。"""
    try:
        import modelscope  # type: ignore  # noqa: F401
        return True
    except Exception:
        pass

    # 尝试通过 pip 安装
    print("检测到未安装 `modelscope`，尝试通过 pip 安装...")
    py = sys.executable
    rc = run([py, "-m", "pip", "install", "--upgrade", "modelscope"])
    if rc == 0:
        try:
            import modelscope
            return True
        except Exception:
            return False
    return False


def snapshot_download_model(model_id: str) -> str:
    """使用 ModelScope 的 snapshot_download 下载模型并返回本地路径。"""
    # 可选：避免在仅下载场景中触发无意义的 CUDA 设备探测告警
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\\.cuda")

    # 延迟导入以便先屏蔽告警
    from modelscope.hub.snapshot_download import snapshot_download

    # 允许使用默认缓存目录（~/.cache/modelscope），也可通过环境变量覆盖
    cache_dir = os.environ.get("MODELSCOPE_CACHE_DIR")  # 如设置将使用该目录
    local_path = snapshot_download(model_id=model_id, cache_dir=cache_dir)
    return local_path


def main(argv: list[str]) -> int:
    model = argv[1] if len(argv) > 1 else DEFAULT_MODEL
    print(f"开始下载模型：{model}")

    if not ensure_modelscope_installed():
        print()
        print("未能自动安装 `modelscope`。请手动安装：")
        print(f"  {sys.executable} -m pip install --upgrade modelscope")
        return 2

    try:
        local_path = snapshot_download_model(model)
        print(f"模型下载完成，已保存到：{local_path}")
        return 0
    except Exception as e: 
        print(f"具体错误：{e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
