"""Verify moving-camera realtime detection overlay in a local browser."""

import json
import sys
import time
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
TASK_ID = "c61e9bf8a2df"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
ARTIFACT_DIR = REPO / "evaluation" / "realtime-detection"


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    ply_url = f"http://127.0.0.1:5173/api/result/{TASK_ID}/scene.ply"
    url = (
        "http://127.0.0.1:5173/splat-editor/index.html"
        f"?load={quote(ply_url, safe='')}&filename=scene.ply&task_id={TASK_ID}"
    )
    requests = 0
    errors: list[str] = []
    started = time.perf_counter()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=str(EDGE),
            headless=True,
            args=["--enable-webgl", "--ignore-gpu-blocklist", "--use-angle=swiftshader"],
        )
        context = browser.new_context(
            viewport={"width": 1280, "height": 800}, service_workers="block"
        )
        context.add_init_script("localStorage.setItem('realtime-detection-enabled', 'true')")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(f"pageerror: {error}"))
        page.on("console", lambda message: errors.append(f"console {message.type}: {message.text}") if message.type == "error" else None)

        def fulfill_detection(route):
            nonlocal requests
            requests += 1
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "frame_id": requests,
                        "width": 640,
                        "height": 360,
                        "inference_ms": 25.0,
                        "detections": [
                            {"category": "bottle", "score": 0.88, "bbox": [0.3, 0.2, 0.5, 0.75]}
                        ],
                    }
                ),
            )

        page.route("**/api/realtime/detect", fulfill_detection)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.locator("#canvas").wait_for(state="visible", timeout=30_000)
        page.wait_for_timeout(8_000)
        offscreen_probe = page.evaluate("""async () => {
            try {
                const value = await window.scene.events.invoke('render.offscreen', 640, 360);
                return { length: value?.length ?? -1 };
            } catch (error) {
                return { error: String(error) };
            }
        }""")
        page.evaluate("window.__triggerRealtimeDetection()")
        try:
            page.locator(".realtime-detection-box").wait_for(timeout=20_000)
        except PlaywrightTimeoutError:
            print(json.dumps({
                "requests": requests,
                "toggle": page.locator(".realtime-detection-toggle").inner_text(),
                "toggle_title": page.locator(".realtime-detection-toggle").get_attribute("title"),
                "active_tool": page.evaluate("window.scene.events.invoke('tool.active')"),
                "document_hidden": page.evaluate("document.hidden"),
                "enabled_storage": page.evaluate("localStorage.getItem('realtime-detection-enabled')"),
                "offscreen_probe": offscreen_probe,
                "errors": errors,
            }, ensure_ascii=False, indent=2))
            raise
        label = page.locator(".realtime-detection-label").inner_text()
        page.screenshot(path=str(ARTIFACT_DIR / "01_moving_detection.png"))

        page.locator(".realtime-detection-toggle").click()
        page.locator(".realtime-detection-box").wait_for(state="detached", timeout=3_000)
        page.wait_for_function(
            "localStorage.getItem('realtime-detection-enabled') === 'false'"
        )
        disabled = page.locator(".realtime-detection-toggle").inner_text()
        disabled_storage = page.evaluate("localStorage.getItem('realtime-detection-enabled')")
        page.locator(".realtime-detection-toggle").click()
        page.evaluate("window.__triggerRealtimeDetection()")
        page.locator(".realtime-detection-box").wait_for(timeout=10_000)

        page.locator("#bottom-toolbar-segmentation").click()
        page.locator("#segmentation-tool.active").wait_for(timeout=5_000)
        page.locator(".realtime-detection-box").wait_for(state="detached", timeout=3_000)
        browser.close()

    result = {
        "pass": requests >= 2 and "瓶子" in label and disabled_storage == "false" and not errors,
        "requests": requests,
        "label": label,
        "disabled_status": disabled,
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    (ARTIFACT_DIR / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
