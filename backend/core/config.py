"""项目配置常量。"""

import os
from pathlib import Path

# 项目根目录: ZipSplat-Object-Reconstruction-Demo/
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv() -> None:
    """轻量读取项目根目录 .env(不引入 python-dotenv 依赖)。

    只填充环境变量未设置的项,已有环境变量优先。
    """
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
TASK_DIR = DATA_DIR / "tasks"
OUTPUT_DIR = DATA_DIR / "outputs"
LAYER_DIR = DATA_DIR / "layers"

# 允许的图片格式
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 上传限制
MAX_UPLOAD_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
# 张数放开到 200:重建前由视图选择自动挑出最优组合(超出部分自动筛选)
MAX_FILE_COUNT = 200

# 视频上传限制
MAX_VIDEO_SIZE_BYTES = 500 * 1024 * 1024  # 每段视频 500 MB
MAX_VIDEO_COUNT = 10
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# 经固定样本对照实验确认的重建参数
DEFAULT_NUM_VIEWS = 6
SCENE_ALPHA_THRESHOLD = 0.02
SCENE_OUTLIER_PERCENTILE = 1
SPLAT_SCALE_FACTOR = 1.0

# SAM 2.1 Hiera Tiny。权重需单独下载，不入版本库。
SAM2_CHECKPOINT = Path(
    os.environ.get("SAM2_CHECKPOINT", PROJECT_ROOT / "models" / "sam2.1_hiera_tiny.pt")
)
SAM2_MODEL_CONFIG = os.environ.get("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_t.yaml")
SEGMENTATION_SESSION_TTL_SECONDS = 10 * 60
MAX_SEGMENTATION_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SEGMENTATION_IMAGE_SIDE = 4096

# ---- CORS（跨域）配置 ----
# 从环境变量读取允许的前端来源（逗号分隔）；默认仅放行本地开发前端。
# 安全约束：绝不同时开放通配符 "*" 与凭证（Starlette 会反射任意来源并带凭证，等同敞开）。
# 局域网来源（手机/平板经 <PC局域网IP>:5173 访问）用正则放行，避免每次改 IP 白名单。
LAN_ORIGIN_REGEX = r"^http://(\d{1,3}\.){3}\d{1,3}:(5173|3000)$"


def resolve_cors() -> tuple[list[str], bool, str | None]:
    """解析 CORS 配置。

    Returns:
        (origins, allow_credentials, allow_origin_regex)
        - origins: 显式来源白名单（绝不含 "*"）。
        - allow_credentials: 仅当来源为显式白名单（非通配符）时才允许携带凭证。
        - allow_origin_regex: 局域网 IP 来源的正则（本地 demo 场景放行手机访问）。
    """
    env = os.environ.get("ZIPSPLAT_CORS_ORIGINS", "")
    origins = [o.strip() for o in env.split(",") if o.strip()]
    if not origins:
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
        ]
    allow_credentials = "*" not in origins
    return origins, allow_credentials, LAN_ORIGIN_REGEX
