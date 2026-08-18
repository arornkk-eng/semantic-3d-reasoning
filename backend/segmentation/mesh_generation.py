"""Run isolated Open3D TSDF conversion for one persistent semantic layer."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from backend.core.config import PROJECT_ROOT
from backend.storage.file_manager import get_output_path
from backend.storage.layer_store import get_layer_metadata


class LayerMeshError(RuntimeError):
    pass


def generate_layer_mesh(task_id: str, layer_id: str) -> tuple[Path, dict]:
    stored = get_layer_metadata(task_id, layer_id)
    if stored is None:
        raise KeyError(layer_id)
    layer, directory = stored
    source = get_output_path(task_id, "scene.ply")
    if source is None or not source.is_file():
        raise LayerMeshError("任务 scene.ply 不存在")
    entries = layer.get("gaussian_indices", [])
    if len(entries) != 1 or entries[0].get("source_index") != 0:
        raise LayerMeshError("当前 TSDF 版本仅支持 scene.ply 的单一 Gaussian 源")
    indices = (directory / str(entries[0].get("file", ""))).resolve()
    if indices.parent != directory.resolve() or not indices.is_file():
        raise LayerMeshError("图层 Gaussian 索引不存在")

    default_python = PROJECT_ROOT / "venv-open3d" / "Scripts" / "python.exe"
    python = Path(os.environ.get("OPEN3D_PYTHON", default_python))
    if not python.is_file():
        raise LayerMeshError("Open3D Python 3.12 环境未安装")
    output = directory / "visual_mesh.ply"
    collision_output = directory / "collision_mesh.ply"
    command = [
        str(python),
        str(PROJECT_ROOT / "tools" / "open3d_layer_tsdf.py"),
        str(source),
        str(indices),
        str(output),
        "--views",
        "24",
        "--resolution",
        "512",
        "--collision-output",
        str(collision_output),
        "--voxel-divisor",
        "160",
        "--collision-triangles",
        "5000",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LayerMeshError("Mesh 生成超过 5 分钟，已终止") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()[-1:] or completed.stdout.strip().splitlines()[-1:]
        raise LayerMeshError(detail[0] if detail else "Open3D TSDF 转换失败")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise LayerMeshError("Open3D 返回了无效结果") from exc
    if not output.is_file() or output.stat().st_size == 0:
        raise LayerMeshError("Open3D 未生成 Mesh 文件")
    if not collision_output.is_file() or collision_output.stat().st_size == 0:
        raise LayerMeshError("Open3D 未生成碰撞候选 Mesh 文件")
    report.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "visual_mesh_file": output.name,
            "collision_mesh_file": collision_output.name,
            "report_version": 1,
        }
    )
    report_path = directory / "mesh-report.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    return output, report
