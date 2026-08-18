"""Run a real browser segmentation flow against local frontend and backend."""

import argparse
import json
import sys
import time
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "evaluation" / "segmentation" / "browser"
DEFAULT_TASK_ID = "c61e9bf8a2df"
DEFAULT_BASE_URL = "http://127.0.0.1:5173"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--generate-mesh", action="store_true")
    parser.add_argument("--replace-cuboid", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task_id = args.task_id
    base_url = args.base_url.rstrip("/")
    artifact_dir = ARTIFACT_DIR / task_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    ply_url = f"{base_url}/api/result/{task_id}/scene.ply"
    editor_url = (
        f"{base_url}/splat-editor/index.html"
        f"?load={quote(ply_url, safe='')}&filename=scene.ply&task_id={task_id}"
    )
    events: list[str] = []
    started = time.perf_counter()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE),
            headless=True,
            args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            service_workers="block",
        )
        page = context.new_page()

        def save_semantic_input(request) -> None:
            if not request.url.endswith("/api/semantic/predict"):
                return
            body = request.post_data_buffer
            content_type = request.headers.get("content-type", "")
            if not body or "multipart/form-data" not in content_type:
                return
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode() + body
            )
            for part in message.iter_parts():
                if part.get_param("name", header="content-disposition") == "image":
                    (artifact_dir / "semantic_input.png").write_bytes(part.get_payload(decode=True))

        page.on("request", save_semantic_input)
        page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
        page.on("pageerror", lambda error: events.append(f"pageerror: {error}"))
        page.goto(editor_url, wait_until="domcontentloaded", timeout=30_000)
        page.locator("#canvas").wait_for(state="visible", timeout=30_000)
        page.wait_for_timeout(10_000)
        page.screenshot(path=str(artifact_dir / "01_scene.png"))

        page.locator("#bottom-toolbar-segmentation").click()
        page.locator("#segmentation-tool.active").wait_for(timeout=5_000)
        page.locator("#segmentation-tool.busy").wait_for(state="hidden", timeout=120_000)
        status = page.locator('#segmentation-tool [data-role="status"]').inner_text()
        page.screenshot(path=str(artifact_dir / "02_mask.png"))
        instance_count = page.locator('.semantic-list [data-action="toggle"]').count()
        projected_status = None
        refined_status = None
        confirmation_status = None
        representation_before = None
        representation_after = None
        scene_layer_names = []
        if instance_count:
            page.locator('[data-action="project3d"]').click()
            page.locator("#segmentation-tool.busy").wait_for(state="hidden", timeout=120_000)
            projected_status = page.locator(
                '#segmentation-tool [data-role="status"]'
            ).inner_text()
            page.locator('[data-action="refine3d"]').click()
            page.locator("#segmentation-tool.busy").wait_for(state="hidden", timeout=120_000)
            refined_status = page.locator(
                '#segmentation-tool [data-role="status"]'
            ).inner_text()
            if args.replace_cuboid:
                page.once("dialog", lambda dialog: dialog.accept())
                page.locator('[data-action="cuboid3d"]').click()
                page.locator("#segmentation-tool.busy").wait_for(
                    state="hidden", timeout=120_000
                )
                confirmation_status = page.locator(
                    '#segmentation-tool [data-role="status"]'
                ).inner_text()
                toggle = page.locator('[data-action="toggle-representation"]')
                toggle.wait_for(timeout=5_000)
                representation_before = toggle.inner_text()
                scene_layer_names = page.locator('.splat-item-name').all_inner_texts()
                toggle.click()
                page.wait_for_timeout(500)
                representation_after = toggle.inner_text()
                page.screenshot(path=str(artifact_dir / "04_original.png"))
                toggle.click()
            else:
                page.once("dialog", lambda dialog: dialog.accept())
                page.locator('[data-action="confirm"]').click()
                page.locator("#segmentation-tool.busy").wait_for(
                    state="hidden", timeout=120_000
                )
                confirmation_status = page.locator(
                    '#segmentation-tool [data-role="status"]'
                ).inner_text()
        page.screenshot(path=str(artifact_dir / "03_saved.png"))
        browser.close()

    layer_root = REPO / "data" / "layers" / task_id
    layers = sorted(layer_root.glob("*/layer.json"), key=lambda path: path.stat().st_mtime)
    metadata = json.loads(layers[-1].read_text(encoding="utf-8")) if layers else None
    mask_path = layers[-1].with_name("mask.png") if layers else None
    mesh_path = None
    mesh_headers = None
    if args.generate_mesh and metadata:
        mesh_path = artifact_dir / f"{metadata['layer_id']}-visual-mesh.ply"
        request = Request(
            f"{base_url}/api/tasks/{task_id}/layers/{metadata['layer_id']}/mesh",
            method="POST",
        )
        with urlopen(request, timeout=360) as response:
            mesh_path.write_bytes(response.read())
            mesh_headers = {
                "vertices": response.headers.get("X-Mesh-Vertices"),
                "triangles": response.headers.get("X-Mesh-Triangles"),
                "collision_triangles": response.headers.get("X-Mesh-Collision-Triangles"),
                "safe_for_collision": response.headers.get("X-Mesh-Safe-For-Collision"),
            }
    result = {
        "editor_url": editor_url,
        "status": status,
        "instance_count": instance_count,
        "projected_status": projected_status,
        "refined_status": refined_status,
        "confirmation_status": confirmation_status,
        "representation_before": representation_before,
        "representation_after": representation_after,
        "scene_layer_names": scene_layer_names,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "layer_json": str(layers[-1]) if layers else None,
        "mask_png": str(mask_path) if mask_path else None,
        "mask_exists": bool(mask_path and mask_path.exists()),
        "mesh_path": str(mesh_path) if mesh_path else None,
        "mesh_exists": bool(mesh_path and mesh_path.is_file()),
        "mesh_headers": mesh_headers,
        "metadata": metadata,
        "browser_events": events,
    }
    result_path = artifact_dir / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
