"""Run a real browser segmentation flow against local frontend and backend."""

import json
import sys
import time
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright


REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "evaluation" / "segmentation" / "browser"
TASK_ID = "c61e9bf8a2df"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ply_url = f"http://127.0.0.1:5173/api/result/{TASK_ID}/scene.ply"
    editor_url = (
        "http://127.0.0.1:5173/splat-editor/index.html"
        f"?load={quote(ply_url, safe='')}&filename=scene.ply&task_id={TASK_ID}"
    )
    events: list[str] = []
    started = time.perf_counter()

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE),
            headless=True,
            args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
        )
        page = browser.new_page(viewport={"width": 1280, "height": 800})

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
                    (ARTIFACT_DIR / "semantic_input.png").write_bytes(part.get_payload(decode=True))

        page.on("request", save_semantic_input)
        page.on("console", lambda message: events.append(f"console {message.type}: {message.text}"))
        page.on("pageerror", lambda error: events.append(f"pageerror: {error}"))
        page.goto(editor_url, wait_until="domcontentloaded", timeout=30_000)
        page.locator("#canvas").wait_for(state="visible", timeout=30_000)
        page.wait_for_timeout(10_000)
        page.screenshot(path=str(ARTIFACT_DIR / "01_scene.png"))

        page.locator("#bottom-toolbar-segmentation").click()
        page.locator("#segmentation-tool.active").wait_for(timeout=5_000)
        page.locator("#segmentation-tool.busy").wait_for(state="hidden", timeout=120_000)
        status = page.locator('[data-role="status"]').inner_text()
        page.screenshot(path=str(ARTIFACT_DIR / "02_mask.png"))
        instance_count = page.locator('.semantic-list [data-action="toggle"]').count()
        if instance_count:
            page.once("dialog", lambda dialog: dialog.accept())
            page.locator('[data-action="confirm"]').click()
            page.locator("#segmentation-tool.active").wait_for(state="hidden", timeout=15_000)
        page.screenshot(path=str(ARTIFACT_DIR / "03_saved.png"))
        browser.close()

    layer_root = REPO / "data" / "layers" / TASK_ID
    layers = sorted(layer_root.glob("*/layer.json"), key=lambda path: path.stat().st_mtime)
    metadata = json.loads(layers[-1].read_text(encoding="utf-8")) if layers else None
    mask_path = layers[-1].with_name("mask.png") if layers else None
    result = {
        "editor_url": editor_url,
        "status": status,
        "instance_count": instance_count,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "layer_json": str(layers[-1]) if layers else None,
        "mask_png": str(mask_path) if mask_path else None,
        "mask_exists": bool(mask_path and mask_path.exists()),
        "metadata": metadata,
        "browser_events": events,
    }
    result_path = ARTIFACT_DIR / "result.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
