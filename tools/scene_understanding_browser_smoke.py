"""Browser smoke test for the scene-understanding panel on the real local PLY."""

import json
import struct
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
TASK_ID = "c61e9bf8a2df"
BASE = "http://127.0.0.1:5173"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
OUT = REPO / "evaluation" / "scene-understanding"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = f"{BASE}/api/result/{TASK_ID}/scene.ply"
    url = (
        f"{BASE}/splat-editor/index.html?load={quote(model, safe='')}"
        f"&filename=scene.ply&task_id={TASK_ID}&build=scene-understanding"
    )
    captured: dict = {}
    errors: list[str] = []
    layers = [
        {
            "layer_id": "bottle1",
            "task_id": TASK_ID,
            "name": "瓶子1",
            "mask_url": "/unused",
            "created_at": "2026-08-11T00:00:00Z",
            "category": "bottle",
            "gaussian_indices": [{
                "source_index": 0, "encoding": "uint32-le", "count": 256,
                "vertex_count": 21866, "url": "/__understanding-smoke__/bottle1.u32"
            }],
        },
        {
            "layer_id": "cup1",
            "task_id": TASK_ID,
            "name": "杯子1",
            "mask_url": "/unused",
            "created_at": "2026-08-11T00:00:00Z",
            "category": "cup",
            "gaussian_indices": [{
                "source_index": 0, "encoding": "uint32-le", "count": 256,
                "vertex_count": 21866, "url": "/__understanding-smoke__/cup1.u32"
            }],
        },
    ]

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(EDGE), headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 800}, service_workers="block")
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda message: errors.append(message.text) if message.type == "error" else None)

        def route_api(route, request):
            path = request.url.split("?", 1)[0]
            if path.endswith(f"/api/tasks/{TASK_ID}/layers/cleanup"):
                route.fulfill(
                    status=200, content_type="application/json",
                    body='{"deleted":true,"layer_count":0,"snapshot_count":0}'
                )
            elif path.endswith(f"/api/tasks/{TASK_ID}/layers"):
                route.fulfill(status=200, content_type="application/json", body=json.dumps(layers))
            elif path.endswith(f"/api/tasks/{TASK_ID}/scene-snapshots"):
                if request.method == "GET":
                    route.fulfill(status=200, content_type="application/json", body="[]")
                else:
                    body = request.post_data_json
                    captured.update(body)
                    response = {
                        "snapshot_id": "snapshot1", "task_id": TASK_ID,
                        "name": "视角分析1", "sequence": 1,
                        "created_at": "2026-08-11T00:00:00Z", **body,
                    }
                    route.fulfill(status=200, content_type="application/json", body=json.dumps(response, ensure_ascii=False))
            elif path.endswith("/__understanding-smoke__/bottle1.u32"):
                route.fulfill(status=200, content_type="application/octet-stream", body=struct.pack("<256I", *range(256)))
            elif path.endswith("/__understanding-smoke__/cup1.u32"):
                route.fulfill(status=200, content_type="application/octet-stream", body=struct.pack("<256I", *range(1024, 1280)))
            else:
                route.continue_()

        page.route("**/*", route_api)
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector(".splat-list .splat-item", timeout=120_000)
        page.locator("#bottom-toolbar-scene-understanding").click()
        panel = page.locator("#scene-understanding-tool.active")
        panel.wait_for(timeout=10_000)
        page.wait_for_function(
            "() => !document.querySelector('#scene-understanding-tool')?.classList.contains('busy') && document.querySelectorAll('#scene-understanding-tool [data-action=toggle-layer]').length === 2",
            timeout=10_000,
        )
        panel.locator('[data-action="toggle-layer"]').nth(0).click()
        panel.locator('[data-action="toggle-layer"]').nth(1).click()
        panel.locator('[data-action="analyze"]').click()
        page.wait_for_timeout(3_000)
        status = panel.locator('[data-role="status"]').inner_text()
        if status != "已保存视角分析1":
            diagnostics = page.evaluate("""() => ({
                active: document.querySelector('#scene-understanding-tool')?.className,
                selected: [...document.querySelectorAll('#scene-understanding-tool .layer-choice.selected')].map(x => x.textContent),
                analyzeDisabled: document.querySelector('#scene-understanding-tool [data-action=analyze]')?.disabled,
                status: document.querySelector('#scene-understanding-tool [data-role=status]')?.textContent
            })""")
            raise AssertionError(f"unexpected status={status!r}, diagnostics={diagnostics!r}, captured={captured!r}, errors={errors!r}")
        page.screenshot(path=str(OUT / "scene-understanding.png"))
        assert len(captured.get("objects", [])) == 2
        assert captured["functions"]["bottle"]
        assert captured["functions"]["cup"]
        assert not errors, errors
        result = {"passed": True, "objects": len(captured["objects"]), "relations": len(captured["relations"]), "description": captured["description"]}
        (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
