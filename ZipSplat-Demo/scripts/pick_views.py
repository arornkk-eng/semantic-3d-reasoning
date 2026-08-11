"""自动选择最佳视图组合:质量过滤 → pycolmap SfM → 贪心覆盖选择。

用户上传一堆照片(可能有废片/重复/乱角度),本脚本自动挑出"最好的组合":
1. 质量过滤:模糊、过曝/欠曝、感知哈希去重
2. pycolmap 轻量 SfM:拿到每张图的相机位姿(剔除配准失败的),降采样加速
3. 贪心最远点采样:在球面上均匀覆盖视角,带最小间距约束和清晰度加权

用法:
    python pick_views.py <图片目录> [--num 12] [--out 输出目录] [--sfm-scale 0.5]

输出:打印选中文件名;--out 指定时把选中图片复制过去,并写 poses.npy / cameras.npy
(位姿为 splatfactory 格式,可直接用于 use_priors=True 重建)。
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

# 纯 ASCII 临时工作区,规避中文路径/编译问题
WORK_ROOT = Path("C:/tempdev/pick_views")
WORK_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


# ------------------------- ① 质量过滤 -------------------------

def _dhash(img: np.ndarray, size: int = 8) -> np.ndarray:
    """感知哈希:灰度图缩到 size×size,相邻像素比较,返回 64bit 位图。"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    return np.packbits(diff.flatten())


def quality_filter(images: dict[str, np.ndarray], blur_factor: float = 0.4) -> list[str]:
    """剔除模糊/过曝/欠曝/重复图,返回通过的路径列表(按清晰度降序)。

    规则:
    - 模糊:Laplacian 方差低于全体中位数的 blur_factor 倍
    - 曝光:灰度均值 < 30(欠曝)或 > 225(过曝)
    - 重复:感知哈希汉明距离 < 8,保留清晰的一张
    """
    names = list(images)
    sharpness = {
        n: cv2.Laplacian(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        for n, im in images.items()
    }
    median = float(np.median(list(sharpness.values())))
    passed = []
    for n in names:
        s = sharpness[n]
        if s < median * blur_factor:
            continue  # 模糊
        mean = images[n].mean()
        if mean < 30 or mean > 225:
            continue  # 过曝/欠曝
        passed.append(n)

    # 感知哈希去重
    hashes = {n: _dhash(images[n]) for n in passed}
    keep: list[str] = []
    for n in passed:
        dup = False
        for k in keep:
            if np.count_nonzero(hashes[n] ^ hashes[k]) < 8:
                dup = True
                break
        if not dup:
            keep.append(n)
    keep.sort(key=lambda n: -sharpness[n])
    return keep


# ------------------------- ② 位姿估计 -------------------------

def run_sfm(images: dict[str, np.ndarray], work_dir: Path) -> dict:
    """pycolmap SfM(单线程 CPU SIFT + exhaustive 匹配)。

    Returns: {filename: c2w_4x4} —— 仅含成功注册的图。
    """
    import pycolmap

    # 图片复制到 ASCII 工作目录(绕开中文路径问题)
    img_dir = work_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for n, im in images.items():
        cv2.imwrite(str(img_dir / n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    db = work_dir / "database.db"
    feat = pycolmap.FeatureExtractionOptions()
    feat.use_gpu = False
    feat.num_threads = 1  # 4.1.1 多线程 SIFT 会段错误
    pycolmap.extract_features(
        database_path=str(db),
        image_path=str(img_dir),
        image_names=list(images),
        extraction_options=feat,
    )
    pycolmap.match_exhaustive(str(db), pycolmap.FeatureMatchingOptions())

    recon_dir = work_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(
        str(db), str(img_dir), str(recon_dir), pycolmap.IncrementalPipelineOptions()
    )
    if not reconstructions:
        return {}
    recon = max(reconstructions.values(), key=lambda item: item.num_reg_images())

    poses = {}
    for name, img in recon.images.items():
        c2w = np.linalg.inv(img.cam_from_world.matrix())  # 4x4
        poses[name] = c2w
    return poses


# ------------------------- ③ 贪心视图选择 -------------------------

def _view_dir(c2w: np.ndarray) -> np.ndarray:
    """相机光轴方向(世界系单位向量)= c2w 旋转矩阵的第三列。"""
    return c2w[:3, 2] / np.linalg.norm(c2w[:3, 2])


def select_views(
    poses: dict, sharpness: dict, num: int, min_gap_deg: float = 8.0
) -> list[str]:
    """贪心最远点采样:每次选"离已选集最远"且间距达标的图。

    - 种子:清晰度最高的
    - 每次迭代:对未选图,计算与已选集的最小角距;
      跳过角距 < min_gap(太近=冗余);
      选角距最大的(覆盖最互补),角距相近时清晰度优先
    """
    names = list(poses)
    if not names:
        return []
    dirs = {n: _view_dir(poses[n]) for n in names}

    chosen = [max(names, key=lambda n: sharpness.get(n, 0))]
    while len(chosen) < num and len(chosen) < len(names):
        best, best_score = None, -1.0
        for n in names:
            if n in chosen:
                continue
            cos = max(float(dirs[n] @ dirs[c]) for c in chosen)
            angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            if angle < min_gap_deg:
                continue
            w = sharpness.get(n, 0) / max(sharpness.values(), default=1)
            score = angle + 20.0 * w  # 角距为主,清晰度为副
            if score > best_score:
                best, best_score = n, score
        if best is None:
            break  # 剩余全部太近,结束
        chosen.append(best)
    return chosen


def select_best_views(
    input_dir: Path, num: int = 12, sfm_scale: float = 0.5
) -> dict:
    """Callable view-selection API used by the backend reconstruction runner."""
    input_dir = Path(input_dir)
    images: dict[str, np.ndarray] = {}
    sharpness: dict[str, float] = {}
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        encoded = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        images[path.name] = image
        sharpness[path.name] = float(
            cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
        )

    passed = quality_filter(images)
    if not passed:
        return {
            "chosen": [],
            "method": "quality_filter_empty",
            "total": len(images),
            "passed": 0,
            "registered": 0,
        }

    sfm_images = {
        name: cv2.resize(
            images[name], None, fx=sfm_scale, fy=sfm_scale, interpolation=cv2.INTER_AREA
        )
        for name in passed
    }
    try:
        with tempfile.TemporaryDirectory(dir=str(WORK_ROOT)) as tmp:
            poses = run_sfm(sfm_images, Path(tmp))
    except Exception:
        poses = {}
    if poses:
        chosen = select_views(poses, sharpness, num)
        method = "sfm_farthest_point"
    else:
        chosen = passed[:num]
        method = "sharpness_fallback"
    return {
        "chosen": chosen,
        "method": method,
        "total": len(images),
        "passed": len(passed),
        "registered": len(poses),
    }


# ------------------------- main -------------------------

def main():
    parser = argparse.ArgumentParser(description="自动选择最佳视图组合")
    parser.add_argument("input", type=Path, help="图片目录")
    parser.add_argument("--num", type=int, default=12, help="目标视图数")
    parser.add_argument("--out", type=Path, default=None, help="输出目录(复制选中图+位姿)")
    parser.add_argument("--sfm-scale", type=float, default=0.5,
                        help="SfM 降采样比例(仅影响位姿精度,加速)")
    args = parser.parse_args()

    # 读图 + 记录清晰度
    images, sharpness = {}, {}
    for p in sorted(args.input.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if im is None:
            continue
        images[p.name] = im
        sharpness[p.name] = cv2.Laplacian(
            cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
    print(f"读取 {len(images)} 张")

    # ① 质量过滤
    passed = quality_filter(images)
    print(f"质量过滤: {len(images)} -> {len(passed)} 张"
          f"(剔除模糊/过曝/重复)")
    if not passed:
        print("全部被过滤,退出")
        return

    # ② SfM(降采样加速)
    sfm_imgs = {
        n: cv2.resize(images[n], None, fx=args.sfm_scale, fy=args.sfm_scale,
                      interpolation=cv2.INTER_AREA)
        for n in passed
    }
    with tempfile.TemporaryDirectory(dir=str(WORK_ROOT)) as tmp:
        print("运行 pycolmap SfM(单线程,可能需要几分钟)...")
        poses = run_sfm(sfm_imgs, Path(tmp))
    print(f"SfM 注册: {len(poses)}/{len(passed)} 张")
    if not poses:
        print("SfM 失败,回退:按清晰度直接取前 N 张")
        chosen = passed[: args.num]
    else:
        # ③ 贪心选择
        chosen = select_views(poses, sharpness, args.num)
        print(f"视图选择: {len(chosen)} 张")

    print("\n===== 选中组合 =====")
    for i, n in enumerate(chosen):
        ang = ""
        if n in poses:
            d = _view_dir(poses[n])
            ang = f" 方位角={np.degrees(np.arctan2(d[0], d[2])):.0f}°"
        print(f"  {i + 1:2d}. {n}{ang}  清晰度={sharpness[n]:.0f}")

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for n in chosen:
            shutil.copy2(args.input / n, out / n)
        if poses:
            # 导出 splatfactory 格式位姿(cameras 用降采样图尺寸,仅参考)
            keep = [n for n in chosen if n in poses]
            if keep:
                cam = np.zeros((len(keep), 6), dtype=np.float32)
                pos = np.zeros((len(keep), 12), dtype=np.float32)
                for i, n in enumerate(keep):
                    c2w = poses[n]
                    pos[i] = c2w[:3].flatten()
                    cam[i] = [images[n].shape[1], images[n].shape[0],
                              images[n].shape[1] * 0.8, images[n].shape[1] * 0.8,
                              images[n].shape[1] / 2, images[n].shape[0] / 2]
                np.save(out / "poses.npy", pos)
                np.save(out / "cameras.npy", cam)
                print(f"位姿已导出: {out / 'poses.npy'} ({len(keep)} 张)")
        print(f"输出目录: {out}")


if __name__ == "__main__":
    main()
