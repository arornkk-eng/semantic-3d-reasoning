"""Browser smoke test for persistent 3D semantic labels."""

import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
TASK_ID = "c61e9bf8a2df"
BASE = "http://127.0.0.1:5173"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
OUT = REPO / "evaluation" / "semantic-labels"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = f"{BASE}/api/result/{TASK_ID}/scene.ply"
    url = (
        f"{BASE}/splat-editor/index.html?load={quote(model, safe='')}"
        f"&filename=scene.ply&task_id={TASK_ID}&build=semantic-labels"
    )
    errors: list[str] = []
    dialogs: list[str] = []
    with urlopen(f"{BASE}/api/tasks/{TASK_ID}/layers", timeout=10) as response:
        persisted_layers = json.load(response)
    assert persisted_layers
    session_layer = {
        **persisted_layers[0],
        "layer_id": "session-label-smoke",
        "name": "本次会话图层",
    }
    add_session_layer = False
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=str(EDGE), headless=True)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800}, service_workers="block"
        )
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()))
        page.on(
            "console",
            lambda message: errors.append(message.text)
            if message.type == "error"
            else None,
        )

        def route_layers(route, request):
            nonlocal add_session_layer
            path = request.url.split("?", 1)[0]
            if path.endswith(f"/api/tasks/{TASK_ID}/layers/cleanup"):
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"deleted":true,"layer_count":0,"snapshot_count":0}',
                )
            elif path.endswith(f"/api/tasks/{TASK_ID}/layers"):
                layers = [*persisted_layers, session_layer] if add_session_layer else persisted_layers
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(layers, ensure_ascii=False),
                )
            else:
                route.continue_()

        page.route("**/*", route_layers)
        page.goto(url, wait_until="domcontentloaded", timeout=120_000)
        page.wait_for_selector(".splat-list .splat-item", timeout=120_000)
        page.locator(".splat-list .splat-item").first.click()
        page.keyboard.press("f")
        page.wait_for_timeout(2_000)
        rows = page.locator("#semantic-layer-label-list .semantic-layer-label-row")
        assert rows.count() == 0
        assert "本次打开" in page.locator("#semantic-layer-label-list").inner_text()
        add_session_layer = True
        page.evaluate("window.scene.events.fire('semantic.layersChanged')")
        page.wait_for_function(
            "() => document.querySelectorAll('#semantic-layer-label-list .semantic-layer-label-row').length === 1",
            timeout=20_000,
        )
        count = rows.count()
        first = rows.first
        first.locator('[data-action="show"]').check()
        label = page.locator(".semantic-3d-label:not([hidden])").first
        try:
            label.wait_for(timeout=30_000)
        except Exception as error:
            state = page.evaluate("""() => ({
                checkbox: document.querySelector('[data-action=show]')?.checked,
                labels: [...document.querySelectorAll('.semantic-3d-label')].map(item => ({
                    hidden: item.hidden, text: item.textContent, style: item.getAttribute('style')
                }))
            })""")
            raise AssertionError(
                f"label did not become visible: state={state!r}, dialogs={dialogs!r}, errors={errors!r}"
            ) from error
        label.click()
        card = page.locator(".semantic-label-card:not([hidden])")
        try:
            card.wait_for(timeout=10_000)
        except Exception as error:
            state = page.evaluate("""() => ({
                cards: [...document.querySelectorAll('.semantic-label-card')].map(item => ({
                    hidden: item.hidden, text: item.textContent, style: item.getAttribute('style'),
                    rect: item.getBoundingClientRect().toJSON()
                }))
            })""")
            raise AssertionError(f"card did not open: {state!r}") from error
        card_text = card.inner_text()
        assert "Gaussian" in card_text
        assert "观察次数" in card_text
        first.locator('[data-action="color"]').fill("#ff3366")
        assert "ff3366" in label.get_attribute("style").lower()
        page.screenshot(path=str(OUT / "01_label_card.png"))
        card.locator('[data-action="hide-label"]').click()
        assert first.locator('[data-action="show"]').is_checked() is False
        assert page.locator(".semantic-3d-label:not([hidden])").count() == 0
        result = {
            "passed": True,
            "initial_rows": 0,
            "session_layer_rows": count,
            "card_text": card_text,
            "errors": errors,
        }
        assert not errors, errors
        (OUT / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        browser.close()


if __name__ == "__main__":
    main()
