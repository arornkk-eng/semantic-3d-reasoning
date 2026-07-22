"""项目配置常量。"""

from pathlib import Path

# 项目根目录: ZipSplat-Object-Reconstruction-Demo/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TASK_DIR = DATA_DIR / "tasks"
OUTPUT_DIR = DATA_DIR / "outputs"

# 允许的图片格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 上传限制
MAX_UPLOAD_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
MAX_FILE_COUNT = 50
