"""分块重建 + 融合实验:验证"分块优化再融合"路线(位置/空间相对关系)。

思路:所有图先跑全局 SfM(公共锚:位姿 + 稀疏点云)。
每块 pose-free 重建后,块高斯点云 ↔ SfM 稀疏点云做 7DoF 相似变换配准
(RANSAC 最近邻对应 + Umeyama + ICP 精修),两块都对齐到同一 SfM 系。

阶段:
    stage1 sfm          : 全局 SfM,保存 poses/points3d + 分块方案
    stage2 reconstruct  : 每块独立 ZipSplat 重建
    stage3 align        : 配准 + 指标 + 融合可视化

用法:
    python block_merge_experiment.py --stage sfm --task 8d8bbd6701d0
    python block_merge_experiment.py --stage reconstruct --task 8d8bbd6701d0
    python block_merge_experiment.py --stage align --task 8d8bbd6701d0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ZipSplat-main"))

from backend.core.config import OUTPUT_DIR, PROJECT_ROOT, UPLOAD_DIR  # noqa: E402

# 实验工作目录
WORK = ROOT / "results" / "block_merge"

# 分块方案(块 A/B 的图名,块 B 含 2 张与 A 重叠的图)
# aa297353418e(视频帧 848x480,按拍摄顺序 12/12 切,无显式重叠图 ——
# 相邻帧天然重叠 + 全局 SfM 公共锚,priors 模式两块直接落在同一 SfM 系)
BLOCK_A = ["frame_0001_v0_t0.0s.jpg", "frame_0002_v0_t0.5s.jpg",
           "frame_0003_v0_t1.9s.jpg", "frame_0004_v0_t2.8s.jpg",
           "frame_0005_v0_t3.7s.jpg", "frame_0006_v0_t4.7s.jpg",
           "frame_0007_v0_t5.6s.jpg", "frame_0008_v0_t6.5s.jpg",
           "frame_0009_v0_t7.0s.jpg", "frame_0010_v0_t7.9s.jpg",
           "frame_0011_v0_t8.9s.jpg", "frame_0012_v0_t9.8s.jpg"]
BLOCK_B = ["frame_0013_v0_t10.7s.jpg", "frame_0014_v0_t11.7s.jpg",
           "frame_0015_v0_t13.1s.jpg", "frame_0016_v0_t14.0s.jpg",
           "frame_0017_v0_t14.5s.jpg", "frame_0018_v0_t15.9s.jpg",
           "frame_0019_v0_t16.8s.jpg", "frame_0020_v0_t17.3s.jpg",
           "frame_0021_v0_t18.2s.jpg", "frame_0022_v0_t19.2s.jpg",
           "frame_0023_v0_t20.6s.jpg", "frame_0024_v0_t21.5s.jpg"]


# ------------------------- stage1: 全局 SfM -------------------------

def _run_sfm(images: dict[str, np.ndarray], work_dir: Path):
    """pycolmap SfM(与 pick_views.run_sfm 同参数),额外导出稀疏点云。"""
    import pycolmap

    img_dir = work_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for n, im in images.items():
        cv2.imwrite(str(img_dir / n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))

    db = work_dir / "database.db"
    feat = pycolmap.FeatureExtractionOptions()
    feat.use_gpu = False
    feat.num_threads = 1
    pycolmap.extract_features(
        database_path=str(db), image_path=str(img_dir),
        image_names=list(images), extraction_options=feat,
    )
    pycolmap.match_exhaustive(str(db), pycolmap.FeatureMatchingOptions())
    recon_dir = work_dir / "recon"
    recon_dir.mkdir(parents=True, exist_ok=True)
    result = pycolmap.incremental_mapping(str(db), str(img_dir), str(recon_dir),
                                          pycolmap.IncrementalPipelineOptions())
    if isinstance(result, dict) and result:
        recon = max(result.values(), key=lambda r: r.num_reg_images())
    else:
        recon = pycolmap.Reconstruction(str(recon_dir))
    return recon


def stage_sfm(task_id: str) -> None:
    ext = {".jpg", ".jpeg", ".png"}
    paths = sorted(
        [p for p in (UPLOAD_DIR / task_id).iterdir() if p.suffix.lower() in ext],
        key=lambda p: p.name,
    )
    images = {p.name: cv2.imread(str(p), cv2.IMREAD_COLOR)[..., ::-1] for p in paths}
    print(f"[sfm] 输入 {len(images)} 张: {sorted(images)}", flush=True)

    work = WORK / task_id
    recon = _run_sfm(images, work / "sfm")
    n_reg = recon.num_reg_images()
    print(f"[sfm] SfM 注册 {n_reg}/{len(images)}", flush=True)
    if n_reg < len(images):
        print(f"[sfm] 警告: 未注册的图会被丢弃,分块将受影响", flush=True)

    # 位姿 {name: c2w 4x4}(未注册图跳过)
    # 注意:pycolmap 的 cam_from_world() 是"世界→相机"(w2c,COLMAP 命名);
    # zipsplat Pose 需要 c2w(相机在世界的位置) → 取 inverse。
    poses: dict[str, np.ndarray] = {}
    for img_id, img in recon.images.items():
        wfc = img.cam_from_world()
        if wfc is None:
            continue
        m = np.asarray(wfc.inverse().matrix(), dtype=float)
        if m.shape == (3, 4):
            m = np.vstack([m, [0, 0, 0, 1]])
        poses[img.name] = m
    # 稀疏点云 (N,3) + 颜色 (N,3),只留观测数 >= 2 的点
    pts, cols = [], []
    for p in recon.points3D.values():
        if p.track.length() >= 2:
            pts.append(p.xyz)
            cols.append(p.color)
    pts = np.asarray(pts, dtype=np.float32)
    cols = np.asarray(cols, dtype=np.uint8) if cols else np.zeros((0, 3), np.uint8)
    print(f"[sfm] 稀疏点 {len(pts)}, 注册图 {len(poses)}", flush=True)

    # 内参 {name: {K, w, h, k}}(priors 模式喂给 ZipSplat 的相机)
    cams: dict[str, dict] = {}
    for img_id, img in recon.images.items():
        cam = img.camera
        K = np.asarray(cam.calibration_matrix(), dtype=float)
        # SIMPLE_RADIAL 的畸变系数 k(渲染对比时对 GT 去畸变用)
        k = float(cam.params[3]) if cam.model_name == "SIMPLE_RADIAL" else 0.0
        cams[img.name] = {
            "K": K.tolist(),
            "w": int(cam.width),
            "h": int(cam.height),
            "k": k,
        }

    (WORK / task_id).mkdir(parents=True, exist_ok=True)
    with open(WORK / task_id / "poses.json", "w", encoding="utf-8") as f:
        json.dump({k: v.tolist() for k, v in poses.items()}, f)
    with open(WORK / task_id / "cameras.json", "w", encoding="utf-8") as f:
        json.dump(cams, f)
    np.save(WORK / task_id / "points3d.npy", pts)
    np.save(WORK / task_id / "points3d_color.npy", cols)
    plan = {"block_a": BLOCK_A, "block_b": BLOCK_B}
    (WORK / task_id / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    print(f"[sfm] 完成: poses.json / points3d.npy / plan.json", flush=True)


# ------------------------- stage2: 分块重建 -------------------------

def _reconstruct_block(images: dict[str, np.ndarray], tag: str, work_dir: Path,
                       use_priors: bool = False) -> None:
    """分块重建。

    use_priors=False: pose-free,输出块内自建世界系。
    use_priors=True: 喂入全局 SfM 的 c2w + K,输出**直接落在 SfM 系**,
    多块融合零配准 —— 这是"分块优化再融合"的正解(坐标天然一致)。
    """
    import torch
    from zipsplat import ZipSplat
    from zipsplat.camera import Camera
    from zipsplat.pose import Pose
    from zipsplat.utils import to_tensor

    model = ZipSplat(weights="zipsplat").cuda().eval()
    names = list(images)
    if use_priors:
        from zipsplat.utils import to_square

        pose_json = json.loads((work_dir / "poses.json").read_text(encoding="utf-8"))
        cam_json = json.loads((work_dir / "cameras.json").read_text(encoding="utf-8"))
        # 横竖拍混拍 → 图与 K 都在外侧修正到 252² 系(喂 252² 图时
        # _prepare_inputs 的 crop/scale 是 no-op,必须自己先修 K)
        imgs, Ks = [], []
        for n in names:
            imgs.append(to_square(to_tensor(images[n])))
            K = np.asarray(cam_json[n]["K"], dtype=np.float32)
            w0, h0 = cam_json[n]["w"], cam_json[n]["h"]
            side = min(w0, h0)
            s = 252.0 / side
            Kc = np.array([[K[0, 0] * s, 0, (K[0, 2] - (w0 - side) / 2.0) * s],
                           [0, K[1, 1] * s, (K[1, 2] - (h0 - side) / 2.0) * s],
                           [0, 0, 1]], dtype=np.float32)
            Ks.append(Kc)
        imgs = torch.stack(imgs)
        c2ws = torch.stack([torch.tensor(pose_json[n], dtype=torch.float32) for n in names])
        Ks = torch.from_numpy(np.stack(Ks))
        cameras = Camera.from_K(Ks, w=252, h=252)
        poses = Pose.from_4x4mat(c2ws)
        with torch.no_grad():
            gaussians = model(imgs, cameras=cameras, poses=poses,
                              use_priors=True, compression=1.0)[0]
        print(f"[recon] {tag}: priors 模式(SfM 系) {len(images)} 张", flush=True)
    else:
        imgs = [to_tensor(im).contiguous() for im in images.values()]
        with torch.no_grad():
            gaussians = model(imgs, compression=1.0)[0]
        print(f"[recon] {tag}: pose-free 模式 {len(images)} 张", flush=True)
    num_gs = gaussians.num_gaussians
    out = work_dir / tag
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "means": gaussians.means.detach().cpu(),
            "scales": gaussians.scales.detach().cpu(),
            "quats": gaussians.quats.detach().cpu(),
            "opacities": gaussians.opacities.detach().cpu(),
            "sh_coeffs": gaussians.sh_coeffs.detach().cpu(),
        },
        out / ("gaussians_priors.pt" if use_priors else "gaussians.pt"),
    )
    gaussians.save_ply(str(out / ("scene_priors.ply" if use_priors else "scene.ply")))
    print(f"[recon] {tag}: {len(images)} 张 → {num_gs:,} 高斯", flush=True)
    del model, gaussians
    torch.cuda.empty_cache()


def stage_reconstruct(task_id: str, use_priors: bool = False) -> None:
    import torch

    plan = json.loads((WORK / task_id / "plan.json").read_text(encoding="utf-8"))
    for tag, names in [("block_a", plan["block_a"]), ("block_b", plan["block_b"])]:
        out = WORK / task_id / tag / ("gaussians_priors.pt" if use_priors else "gaussians.pt")
        if out.exists():
            print(f"[recon] {tag} 已存在,跳过", flush=True)
            continue
        images = {
            n: cv2.imread(str(UPLOAD_DIR / task_id / n), cv2.IMREAD_COLOR)[..., ::-1]
            for n in names if (UPLOAD_DIR / task_id / n).exists()
        }
        print(f"[recon] {tag}: {len(images)} 张 {names}", flush=True)
        _reconstruct_block(images, tag, WORK / task_id, use_priors=use_priors)
    print("[recon] 完成", flush=True)


# ------------------------- stage3: 对齐 -------------------------

def _umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Umeyama 相似变换: dst ≈ s * R @ src + t。返回 (R(3,3), t(3,), s)。"""
    s_mean, d_mean = src.mean(0), dst.mean(0)
    s_cent, d_cent = src - s_mean, dst - d_mean
    H = d_cent.T @ s_cent
    U, _, Vt = np.linalg.svd(H)
    R = U @ np.diag([1, 1, np.linalg.det(U @ Vt)]) @ Vt
    s = np.sum(d_cent * (s_cent @ R.T)) / np.sum(s_cent * s_cent)
    t = d_mean - s * R @ s_mean
    return R, t, s


def _apply_sim3(points: np.ndarray, R: np.ndarray, t: np.ndarray, s: float) -> np.ndarray:
    return s * points @ R.T + t


def _build_correspondences(src: np.ndarray, dst: np.ndarray,
                           src_cols: np.ndarray | None, dst_cols: np.ndarray | None,
                           color_thr: float = 60.0) -> tuple[np.ndarray, np.ndarray]:
    """颜色空间双向最近邻一致,得到 src↔dst 的对应集。

    几何空间在坐标系未对齐时最近邻毫无意义(距离被相似变换拉爆),
    但颜色与坐标系无关,是同一点在不同块中的不变量 → 在 RGB 空间
    做互近邻一致(可选颜色距离过滤),错配交给 RANSAC 的几何 inlier 筛。
    无颜色时退回几何空间双向一致(ICP 精修阶段可用)。
    """
    from scipy.spatial import cKDTree

    if src_cols is not None and dst_cols is not None and len(dst_cols):
        s_cols = src_cols.astype(np.float32)
        d_cols = dst_cols.astype(np.float32)
        t_src, t_dst = cKDTree(s_cols), cKDTree(d_cols)
        _, nn_s = t_src.query(d_cols, k=1)  # 每个 dst 点的 src 颜色最近邻
        _, nn_d = t_dst.query(s_cols, k=1)  # 每个 src 点的 dst 颜色最近邻
        ok = nn_d[nn_s] == np.arange(len(dst))  # 颜色互近邻一致
        cd = np.linalg.norm(s_cols[nn_s] - d_cols, axis=1)
        ok &= cd < color_thr
    else:
        t_src, t_dst = cKDTree(src), cKDTree(dst)
        _, nn_s = t_src.query(dst, k=1)
        _, nn_d = t_dst.query(src, k=1)
        ok = nn_d[nn_s] == np.arange(len(dst))
    return nn_s[ok], np.arange(len(dst))[ok]


def _ransac_sim3(src: np.ndarray, dst: np.ndarray, iters: int = 2000,
                 dist_thr: float = 0.05) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """对应集 + RANSAC 相似变换(src→dst)。返回 R, t, s, inlier_mask(dst 侧)。

    调用方先经 _build_correspondences 得到 src[idx] ↔ dst[idx] 的对应对;
    这里直接对 (src, dst) 视作已一一对应的点对集做 RANSAC。
    """
    rng = np.random.default_rng(42)
    best = None
    n = len(dst)
    if n < 4:
        return np.eye(3), np.zeros(3), 1.0, np.zeros(n, bool)
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        R, t, s = _umeyama_sim3(src[idx], dst[idx])
        pred = _apply_sim3(src, R, t, s)
        err = np.linalg.norm(pred - dst, axis=1)
        inl = err < dist_thr
        if best is None or inl.sum() > best[0]:
            best = (int(inl.sum()), R, t, s, err)
    _, R, t, s, err = best
    # 全部 inlier 重估计
    if err is not None and err.size:
        m = err < dist_thr
        if m.sum() >= 3:
            R, t, s = _umeyama_sim3(src[m], dst[m])
    return R, t, s, err < dist_thr if err is not None else np.zeros(n, bool)


def _icp_refine(src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray,
                s: float, iters: int = 30) -> tuple[np.ndarray, np.ndarray, float]:
    """ICP 精修(最近邻迭代 + Umeyama 相似变换),尺度加保护防止发散。"""
    from scipy.spatial import cKDTree

    s0 = s
    for _ in range(iters):
        tree = cKDTree(_apply_sim3(src, R, t, s))
        dists, nn = tree.query(dst, k=1)
        m = dists < np.percentile(dists, 90)  # 只信 90% 最近的点
        if m.sum() < 3:
            break
        R, t, s = _umeyama_sim3(src[nn[m]], dst[m])
        s = float(np.clip(s, 0.3 * s0, 3.0 * s0))  # 尺度保护:防 ICP 把尺度拉爆
    return R, t, s


def stage_align(task_id: str) -> None:
    import torch
    from zipsplat.gaussians import Gaussians

    def _block_pt(tag: str) -> Path:
        """块高斯参数:priors 模式产出 gaussians_priors.pt,pose-free 产出 gaussians.pt。"""
        p = WORK / task_id / tag / "gaussians_priors.pt"
        return p if p.exists() else WORK / task_id / tag / "gaussians.pt"

    pts_sfm = np.load(WORK / task_id / "points3d.npy")
    if len(pts_sfm) < 6:
        print(f"[align] SfM 点太少({len(pts_sfm)}),无法对齐", flush=True)
        return

    # 场景尺度估计(SfM 系):用点云对角线的 ~1% 做距离阈值参考
    diag = float(np.linalg.norm(pts_sfm.max(0) - pts_sfm.min(0)))
    thr = 0.02 * diag
    print(f"[align] SfM 点 {len(pts_sfm)}, 场景对角线 {diag:.2f}, 配准阈值 {thr:.3f}", flush=True)

    transforms = {}
    for tag in ["block_a", "block_b"]:
        params = torch.load(_block_pt(tag),
                            map_location="cpu", weights_only=True)
        g = Gaussians.from_parameters(**params)
        means = g.means.detach().cpu().numpy()
        opac = g.opacities.detach().cpu().numpy().reshape(-1)
        _SH_C0 = 0.28209479177387814
        sh0 = g.sh_coeffs.detach().cpu().numpy()[:, 0] * _SH_C0 + 0.5  # (N,3) 0~1
        keep = opac > 0.05
        pts_all = means[keep]
        cols_all = np.clip(sh0[keep] * 255, 0, 255).astype(np.uint8)
        # 高斯点可能很密(数万),抽到 ~8k 配准(快且够)
        if len(pts_all) > 8000:
            rng = np.random.default_rng(0)
            pts_g_idx = rng.choice(len(pts_all), 8000, replace=False)
            pts_g, cols_g = pts_all[pts_g_idx], cols_all[pts_g_idx]
        else:
            pts_g, cols_g = pts_all, cols_all
        print(f"[align] {tag}: 高斯 {len(means):,} → 配准点 {len(pts_g):,}", flush=True)

        # 配准:中心对齐 + RMS 尺度粗配,再 ICP 精修。
        # 实验发现模型自建系与 SfM 系几乎无旋转差(中心+尺度后 median≈0.5),
        # 颜色引导的 RANSAC 因最近邻 inlier 在"包含关系"下失效,弃用。
        sm_g, sm_s = pts_g.mean(0), pts_sfm.mean(0)
        sg = np.sqrt(np.mean(np.sum((pts_g - sm_g) ** 2, 1)))
        ss = np.sqrt(np.mean(np.sum((pts_sfm - sm_s) ** 2, 1)))
        s_est = ss / sg
        R0 = np.eye(3)
        t0 = sm_s - s_est * sm_g
        print(f"[align] {tag}: 粗配 s={s_est:.3f}(RMS 半径比), 待 ICP 精修", flush=True)
        R, t, s = _icp_refine(pts_g, pts_sfm, R0, t0, s_est)
        transforms[tag] = {"R": R.tolist(), "t": t.tolist(), "s": float(s)}
        # 对齐后残差
        pts_t = _apply_sim3(pts_g, R, t, s)
        from scipy.spatial import cKDTree

        tree = cKDTree(pts_t)
        d, _ = tree.query(pts_sfm, k=1)
        # 尺度/位置诊断:模型系与 SfM 系 bbox
        bbox_g = pts_g.max(0) - pts_g.min(0)
        bbox_t = pts_t.max(0) - pts_t.min(0)
        bbox_s = pts_sfm.max(0) - pts_sfm.min(0)
        print(f"[align] {tag}: s={s:.4f}, 模型系 bbox 对角线 "
              f"{float(np.linalg.norm(bbox_g)):.3f} → 变换后 {float(np.linalg.norm(bbox_t)):.2f}"
              f" (SfM={float(np.linalg.norm(bbox_s)):.2f})", flush=True)
        print(f"[align] {tag}: 与 SfM 点最近邻距离 "
              f"median={np.median(d):.4f} p90={np.percentile(d, 90):.4f}", flush=True)

    # ---- 融合与指标 ----
    merged_pts, merged_cols = [], []
    for tag in ["block_a", "block_b"]:
        params = torch.load(_block_pt(tag),
                            map_location="cpu", weights_only=True)
        g = Gaussians.from_parameters(**params)
        m = g.means.detach().cpu().numpy()
        op = g.opacities.detach().cpu().numpy().reshape(-1)
        keep = op > 0.05
        m, op = m[keep], op[keep]
        R = np.asarray(transforms[tag]["R"]); t = np.asarray(transforms[tag]["t"])
        merged_pts.append(_apply_sim3(m, R, t, transforms[tag]["s"]))
        # 颜色:SH DC → RGB(粗,仅可视化)
        sh0 = g.sh_coeffs.detach().cpu().numpy()[:, 0] * 0.28209479177387814 + 0.5
        merged_cols.append(np.clip(sh0[keep] * 255, 0, 255).astype(np.uint8))
    mpts = np.concatenate(merged_pts)
    mcols = np.concatenate(merged_cols)
    print(f"[align] 融合高斯 {mpts.shape[0]:,}", flush=True)

    # 融合高斯完整导出(可直接在 SuperSplat 打开):means/scales/quats 都做
    # 相似变换 —— means 缩放旋转平移, scales 乘 s, quats 旋转 R
    def _rot_quat(R: np.ndarray) -> np.ndarray:
        R = np.asarray(R, dtype=np.float64)
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0:
            qw = 0.5 * np.sqrt(1 + tr); s2 = 0.25 / qw
            q = np.array([qw, (R[2, 1] - R[1, 2]) * s2, (R[0, 2] - R[2, 0]) * s2,
                          (R[1, 0] - R[0, 1]) * s2])
        else:
            i = int(np.argmax(np.diag(R)))
            j, k = (i + 1) % 3, (i + 2) % 3
            qv = np.zeros(3)
            qv[i] = 0.5 * np.sqrt(max(0.0, 1 + R[i, i] - R[j, j] - R[k, k]))
            qv[j] = (R[j, i] + R[i, j]) / (4 * qv[i]) if qv[i] else 0.0
            qv[k] = (R[k, i] + R[i, k]) / (4 * qv[i]) if qv[i] else 0.0
            q = np.array([qv[j] * (R[k, j] - R[j, k]) / (4 * qv[i]) if qv[i] else 1.0,
                          qv[0], qv[1], qv[2]])
        return q / np.linalg.norm(q)

    def _qmul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    merged = {k: [] for k in
              ["x", "y", "z", "scale_0", "scale_1", "scale_2",
               "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
               "f_dc_0", "f_dc_1", "f_dc_2",
               "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3",
               "f_rest_4", "f_rest_5", "f_rest_6", "f_rest_7", "f_rest_8"]}
    for tag in ["block_a", "block_b"]:
        params = torch.load(_block_pt(tag),
                            map_location="cpu", weights_only=True)
        g = Gaussians.from_parameters(**params)
        R = np.asarray(transforms[tag]["R"]); t = np.asarray(transforms[tag]["t"])
        s = float(transforms[tag]["s"])
        means = (s * g.means.detach().cpu().numpy() @ R.T + t)
        scales = g.scales.detach().cpu().numpy() * s
        qr = _rot_quat(R)
        quats = np.stack([_qmul(qr, q) for q in g.quats.detach().cpu().numpy()])
        op = g.opacities.detach().cpu().numpy().reshape(-1)
        keep = op > 0.05
        sh = g.sh_coeffs.detach().cpu().numpy()  # (N, K, 3)
        sh0 = sh[:, 0]                            # 原始 SH DC
        # 一阶 SH 与 save_ply 相同编码:channel-first (3, K-1) 展开 → f_rest_0..8
        f_rest = sh[:, 1:].transpose(0, 2, 1).reshape(len(sh), -1)  # (N, 9)
        merged["x"].append(means[keep, 0]); merged["y"].append(means[keep, 1])
        merged["z"].append(means[keep, 2])
        # 3DGS PLY 标准:scale 存 log-space(save_ply 同款 clamp+log)
        merged["scale_0"].append(np.log(np.clip(scales[keep, 0], 1e-6, None)))
        merged["scale_1"].append(np.log(np.clip(scales[keep, 1], 1e-6, None)))
        merged["scale_2"].append(np.log(np.clip(scales[keep, 2], 1e-6, None)))
        merged["rot_0"].append(quats[keep, 0]); merged["rot_1"].append(quats[keep, 1])
        merged["rot_2"].append(quats[keep, 2]); merged["rot_3"].append(quats[keep, 3])
        merged["opacity"].append(np.log(np.clip(op[keep], 1e-6, 1 - 1e-6)
                                        / (1 - np.clip(op[keep], 1e-6, 1 - 1e-6))))
        merged["f_dc_0"].append(sh0[keep, 0]); merged["f_dc_1"].append(sh0[keep, 1])
        merged["f_dc_2"].append(sh0[keep, 2])
        for i in range(9):
            merged[f"f_rest_{i}"].append(f_rest[keep, i])
    from plyfile import PlyData, PlyElement
    names = ["x", "y", "z", "scale_0", "scale_1", "scale_2",
             "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
             "f_dc_0", "f_dc_1", "f_dc_2",
             "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3",
             "f_rest_4", "f_rest_5", "f_rest_6", "f_rest_7", "f_rest_8"]
    arr = np.column_stack([np.concatenate(merged[k]) for k in names]).astype(np.float32)
    struct = np.zeros(len(arr), dtype=[(n, "f4") for n in names])
    for i, n in enumerate(names):
        struct[n] = arr[:, i]
    el = PlyElement.describe(struct, "vertex")
    PlyData([el], text=False).write(str(WORK / task_id / "merged_splats.ply"))
    print(f"[align] 已导出融合高斯 → merged_splats.ply (SuperSplat 可打开)", flush=True)

    # 重叠区检验:两块变换后互查最近邻距离(空间相对正确则整块贴近)
    from scipy.spatial import cKDTree

    # 两块独立变换后互查
    res = {}
    for tag, other in [("block_a", "block_b"), ("block_b", "block_a")]:
        params = torch.load(_block_pt(tag),
                            map_location="cpu", weights_only=True)
        g = Gaussians.from_parameters(**params)
        m = g.means.detach().cpu().numpy()
        op = g.opacities.detach().cpu().numpy().reshape(-1)
        m = m[op > 0.05]
        R = np.asarray(transforms[tag]["R"]); t = np.asarray(transforms[tag]["t"])
        pts_t = _apply_sim3(m, R, t, transforms[tag]["s"])
        params = torch.load(_block_pt(other),
                            map_location="cpu", weights_only=True)
        g = Gaussians.from_parameters(**params)
        m2 = g.means.detach().cpu().numpy()
        op2 = g.opacities.detach().cpu().numpy().reshape(-1)
        m2 = m2[op2 > 0.05]
        R2 = np.asarray(transforms[other]["R"]); t2 = np.asarray(transforms[other]["t"])
        pts_o = _apply_sim3(m2, R2, t2, transforms[other]["s"])
        tree = cKDTree(pts_o)
        d, _ = tree.query(pts_t, k=1)
        # 只统计"近邻距离 < 2×thr"的重叠点(避免大片独立区域干扰)
        close = d < 2 * thr
        res[tag] = {
            "overlap_ratio": float(close.mean()),
            "median_dist": float(np.median(d[close])) if close.any() else None,
            "p90_dist": float(np.percentile(d[close], 90)) if close.any() else None,
        }
        print(f"[align] {tag}→{other}: 重叠率 {close.mean():.2%}, "
              f"重叠区最近邻 median={np.median(d[close]):.4f} p90={np.percentile(d[close], 90):.4f}",
              flush=True)

    # 与全量重建参考对比(8d8bbd6701d0 的高斯,变换到 SfM 系)
    ref_path = OUTPUT_DIR / task_id / "gaussians.pt"
    if ref_path.exists():
        params = torch.load(ref_path, map_location="cpu", weights_only=True)
        g = Gaussians.from_parameters(**params)
        mref = g.means.detach().cpu().numpy()
        opref = g.opacities.detach().cpu().numpy().reshape(-1)
        mref = mref[opref > 0.05]
        # 全量重建本身也在自己的世界系:先把它也配准到 SfM 系(同样:中心+RMS 尺度 → ICP)
        smr, sms = mref.mean(0), pts_sfm.mean(0)
        sgr = np.sqrt(np.mean(np.sum((mref - smr) ** 2, 1)))
        ssr = np.sqrt(np.mean(np.sum((pts_sfm - sms) ** 2, 1)))
        s_est = ssr / sgr
        Rf, tf, sf = np.eye(3), sms - s_est * smr, s_est
        Rf, tf, sf = _icp_refine(mref, pts_sfm, Rf, tf, sf)
        mref_t = _apply_sim3(mref, Rf, tf, sf)
        tree = cKDTree(mref_t)
        d, _ = tree.query(mpts, k=1)
        close = d < 2 * thr
        res["vs_full"] = {
            "overlap_ratio": float(close.mean()),
            "median_dist": float(np.median(d[close])) if close.any() else None,
        }
        print(f"[align] 融合 vs 全量参考: 重叠率 {close.mean():.2%}, "
              f"重叠区最近邻 median={np.median(d[close]):.4f}", flush=True)

    # 保存融合点云(PLY 便于 SuperSplat 目检)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(mpts)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    )
    with open(WORK / task_id / "merged.ply", "wb") as f:
        f.write(header.encode())
        verts = np.hstack([mpts.astype(np.float32), mcols]).tobytes()
        f.write(verts)
    (WORK / task_id / "align_result.json").write_text(
        json.dumps({"transforms": transforms, "metrics": res}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[align] 完成 → merged.ply / align_result.json", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="分块融合实验")
    parser.add_argument("--stage", required=True, choices=["sfm", "reconstruct", "align"])
    parser.add_argument("--task", default="8d8bbd6701d0")
    parser.add_argument("--mode", choices=["pose-free", "priors"], default="priors",
                        help="重建模式: pose-free(需配准) / priors(SfM 系,默认)")
    args = parser.parse_args()
    if args.stage == "sfm":
        stage_sfm(args.task)
    elif args.stage == "reconstruct":
        stage_reconstruct(args.task, use_priors=args.mode == "priors")
    else:
        stage_align(args.task)


if __name__ == "__main__":
    main()
