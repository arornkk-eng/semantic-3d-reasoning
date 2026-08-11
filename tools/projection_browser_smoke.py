"""Smoke-test semantic-mask projection in the locally served SuperSplat editor."""

from __future__ import annotations

import argparse
import json
import re
import struct
import sys
import time
import traceback
import urllib.error
import urllib.request
import zlib
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from typing import Any
from urllib.parse import quote

from playwright.sync_api import Page, sync_playwright
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

REPO = Path(__file__).resolve().parents[1]
TASK_ID = "c61e9bf8a2df"
DEFAULT_BASE_URL = "http://127.0.0.1:5173"
DEFAULT_ARTIFACT_DIR = REPO / "evaluation" / "segmentation" / "projection"
VIEWPORT = {"width": 1280, "height": 800}
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")
PROJECTED_STATUS_RE = re.compile(r"^已投射\s+([1-9]\d*)\s+个\s+Gaussian$")
EXPANDED_STATUS_RE = re.compile(r"^种子\s+([1-9]\d*)，扩张\s+\+(\d+)$")


class SmokeFailure(AssertionError):
    """Raised when a required smoke-test condition is not met."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    parser.add_argument("--headed", action="store_true")
    return parser.parse_args()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(payload, checksum) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def white_png(width: int, height: int) -> bytes:
    if not 1 <= width <= 4096 or not 1 <= height <= 4096:
        raise ValueError(f"invalid capture size {width}x{height}")
    row = b"\x00" + b"\xff\xff\xff\xff" * width
    pixels = row * height
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", header)
        + png_chunk(b"IDAT", zlib.compress(pixels, level=6))
        + png_chunk(b"IEND", b"")
    )


def parse_predict_request(request: Any) -> tuple[dict[str, Any], int]:
    content_type = request.headers.get("content-type", "")
    body = request.post_data_buffer
    if not body or "multipart/form-data" not in content_type:
        raise ValueError("semantic predict request is not multipart/form-data")
    if isinstance(body, str):
        body = body.encode()
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
        + body
    )
    metadata: dict[str, Any] | None = None
    image_count = 0
    for part in message.iter_parts():
        field = part.get_param("name", header="content-disposition")
        if field == "image":
            image_count += 1
        elif field == "metadata":
            payload = part.get_payload(decode=True) or b""
            metadata = json.loads(payload.decode(part.get_content_charset() or "utf-8"))
    if metadata is None:
        raise ValueError("semantic predict request has no metadata field")
    return metadata, image_count


def check_url(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read(1)
            return {"url": url, "ok": True, "status": response.status}
    except (OSError, urllib.error.URLError) as error:
        return {"url": url, "ok": False, "error": str(error)}


def parse_count(text: str) -> int | None:
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def selected_count(page: Page) -> int | None:
    values = page.locator("#status-bar .status-bar-stat-value")
    if values.count() < 2:
        return None
    return parse_count(values.nth(1).inner_text())


def wait_for_scene(page: Page, timeout_ms: int) -> None:
    page.locator("#canvas").wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_function(
        """() => {
            const value = document.querySelectorAll(
                '#status-bar .status-bar-stat-value'
            )[0];
            const count = Number((value?.textContent || '').replace(/[^0-9]/g, ''));
            return count > 0;
        }""",
        timeout=timeout_ms,
    )


def wait_for_undo_restored(
    page: Page, initial_splat_items: int, timeout_ms: int
) -> bool:
    try:
        page.wait_for_function(
            """initialItems => {
                const value = document.querySelectorAll(
                    '#status-bar .status-bar-stat-value'
                )[1];
                const text = value?.textContent || '';
                return document.querySelectorAll('.splat-list .splat-item').length ===
                    initialItems && /[0-9]/.test(text) &&
                    Number(text.replace(/[^0-9]/g, '')) === 0;
            }""",
            arg=initial_splat_items,
            timeout=timeout_ms,
        )
        return True
    except PlaywrightTimeoutError:
        return False


def run(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base_url = args.base_url.rstrip("/")
    ply_url = f"{base_url}/api/result/{TASK_ID}/scene.ply"
    editor_url = (
        f"{base_url}/splat-editor/index.html"
        f"?load={quote(ply_url, safe='')}&filename=scene.ply&task_id={TASK_ID}"
    )
    preflight = [check_url(base_url), check_url(ply_url)]
    result: dict[str, Any] = {
        "passed": False,
        "task_id": TASK_ID,
        "editor_url": editor_url,
        "viewport": VIEWPORT,
        "preflight": preflight,
        "screenshots": [],
    }
    result_path = artifact_dir / "result.json"
    if not all(item["ok"] for item in preflight):
        result["error"] = "frontend or scene service is unavailable"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    started = time.perf_counter()
    page_errors: list[str] = []
    console_errors: list[str] = []
    route_errors: list[str] = []
    mask_requests: list[str] = []
    mock: dict[str, Any] = {
        "predict_requests": 0,
        "metadata": None,
        "image_count": 0,
        "capture_width": None,
        "capture_height": None,
        "mask_png": None,
    }
    page: Page | None = None
    browser = None

    try:
        with sync_playwright() as playwright:
            launch_options: dict[str, Any] = {
                "headless": not args.headed,
                "args": [
                    "--enable-webgl",
                    "--ignore-gpu-blocklist",
                    "--use-angle=swiftshader",
                ],
            }
            if EDGE.exists():
                launch_options["executable_path"] = str(EDGE)
            browser = playwright.chromium.launch(**launch_options)
            page = browser.new_page(viewport=VIEWPORT)
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on(
                "console",
                lambda message: console_errors.append(message.text)
                if message.type == "error"
                else None,
            )

            def fulfill_predict(route: Any) -> None:
                try:
                    metadata, image_count = parse_predict_request(route.request)
                    width = int(metadata["capture_width"])
                    height = int(metadata["capture_height"])
                    views = metadata.get("views")
                    if image_count != 3 or not isinstance(views, list) or len(views) != 3:
                        raise ValueError(
                            f"expected 3 captured views, got {image_count} images and "
                            f"{len(views) if isinstance(views, list) else 0} metadata views"
                        )
                    mock.update(
                        {
                            "predict_requests": mock["predict_requests"] + 1,
                            "metadata": metadata,
                            "image_count": image_count,
                            "capture_width": width,
                            "capture_height": height,
                            "mask_png": white_png(width, height),
                        }
                    )
                    mask_urls = [
                        f"/__projection-smoke__/mask-{index}.png"
                        for index in range(3)
                    ]
                    instances = [
                        {
                            "instance_id": "white-cup-1",
                            "category": "cup",
                            "category_zh": "杯子",
                            "instance_index": 1,
                            "score": 1.0,
                            "bbox": [0, 0, width, height],
                            "mask_url": mask_urls[0],
                            "depth_coverage": 1.0,
                            "view_support": 3,
                            "view_count": 3,
                            "view_masks": [
                                {"view_index": index, "mask_url": mask_url}
                                for index, mask_url in enumerate(mask_urls)
                            ],
                        }
                    ]
                    route.fulfill(
                        status=200,
                        content_type="application/json; charset=utf-8",
                        body=json.dumps(
                            {"result_id": "projection-smoke", "instances": instances},
                            ensure_ascii=False,
                        ),
                    )
                except Exception:
                    route_errors.append(traceback.format_exc())
                    route.fulfill(
                        status=500,
                        content_type="application/json",
                        body=json.dumps({"detail": "projection smoke mock failed"}),
                    )

            def fulfill_mask(route: Any) -> None:
                mask = mock.get("mask_png")
                if not isinstance(mask, bytes):
                    route_errors.append("mask requested before predict metadata was parsed")
                    route.fulfill(status=500, body=b"")
                    return
                mask_requests.append(route.request.url)
                route.fulfill(status=200, content_type="image/png", body=mask)

            page.route(re.compile(r"/api/semantic/predict(?:\?|$)"), fulfill_predict)
            page.route(
                re.compile(r"/__projection-smoke__/mask-[0-2]\.png(?:\?|$)"),
                fulfill_mask,
            )
            page.goto(editor_url, wait_until="domcontentloaded", timeout=args.timeout_ms)
            wait_for_scene(page, args.timeout_ms)

            initial_selected = selected_count(page)
            initial_splat_items = page.locator('.splat-list .splat-item').count()
            if initial_splat_items != 1:
                raise SmokeFailure(
                    f"expected 1 initial Splat item, got {initial_splat_items}"
                )
            page.keyboard.press("f")
            page.wait_for_timeout(1_000)
            focused = artifact_dir / "01_focused.png"
            page.screenshot(path=str(focused))
            result["screenshots"].append(str(focused))

            segmentation = page.locator("#bottom-toolbar-segmentation")
            segmentation.click(timeout=10_000)
            tool = page.locator("#segmentation-tool")
            tool.locator('.semantic-list [data-action="toggle"]').first.wait_for(
                state="visible", timeout=args.timeout_ms
            )
            page.locator("#segmentation-tool.busy").wait_for(
                state="hidden", timeout=args.timeout_ms
            )
            instance_count = tool.locator(
                '.semantic-list [data-action="toggle"]'
            ).count()
            if instance_count != 1:
                raise SmokeFailure(f"expected 1 mocked instance, got {instance_count}")
            if route_errors:
                raise SmokeFailure(route_errors[-1])
            masks = artifact_dir / "02_three_white_masks.png"
            page.screenshot(path=str(masks))
            result["screenshots"].append(str(masks))

            project_button = tool.locator('[data-action="project3d"]')
            if project_button.count() != 1:
                raise SmokeFailure("served SuperSplat build has no 投射到3D button")
            project_button.click(timeout=10_000)
            page.wait_for_function(
                r"""() => /^已投射\s+[1-9][0-9]*\s+个\s+Gaussian$/.test(
                    document.querySelector('#segmentation-tool [data-role="status"]')
                        ?.textContent || ''
                )""",
                timeout=args.timeout_ms,
            )
            status = tool.locator('[data-role="status"]').inner_text().strip()
            match = PROJECTED_STATUS_RE.fullmatch(status)
            if not match:
                raise SmokeFailure(f"unexpected projection status: {status!r}")
            projected_count = int(match.group(1))
            if projected_count <= 0:
                raise SmokeFailure(f"projection selected {projected_count} Gaussians")
            projected_selected = selected_count(page)
            projected = artifact_dir / "03_projected.png"
            page.screenshot(path=str(projected))
            result["screenshots"].append(str(projected))

            expand_button = tool.locator('[data-action="expand3d"]')
            if expand_button.count() != 1:
                raise SmokeFailure("served SuperSplat build has no 补全3D button")
            expand_button.click(timeout=10_000)
            page.wait_for_function(
                r"""() => /^种子\s+[1-9][0-9]*，扩张\s+\+[0-9]+$/.test(
                    document.querySelector('#segmentation-tool [data-role="status"]')
                        ?.textContent || ''
                )""",
                timeout=args.timeout_ms,
            )
            expanded_status = tool.locator('[data-role="status"]').inner_text().strip()
            expanded_match = EXPANDED_STATUS_RE.fullmatch(expanded_status)
            if not expanded_match:
                raise SmokeFailure(f"unexpected expansion status: {expanded_status!r}")
            expanded_seed_count = int(expanded_match.group(1))
            expanded_added_count = int(expanded_match.group(2))
            expanded = artifact_dir / "04_expanded.png"
            page.screenshot(path=str(expanded))
            result["screenshots"].append(str(expanded))

            refine_button = tool.locator('[data-action="refine3d"]')
            if refine_button.count() != 1:
                raise SmokeFailure("served SuperSplat build has no 精细补全 button")
            refine_button.click(timeout=10_000)
            page.wait_for_function(
                r"""() => /^精细种子\s+[1-9][0-9]*，open3d-style-scipy\s+\+[0-9]+$/.test(
                    document.querySelector('#segmentation-tool [data-role="status"]')
                        ?.textContent || ''
                )""",
                timeout=args.timeout_ms,
            )
            refined_status = tool.locator('[data-role="status"]').inner_text().strip()
            refined = artifact_dir / "05_refined.png"
            page.screenshot(path=str(refined))
            result["screenshots"].append(str(refined))

            separate_button = tool.locator('[data-action="separate3d"]')
            if separate_button.count() != 1:
                raise SmokeFailure("served SuperSplat build has no 生成独立3D图层 button")
            separate_button.click(timeout=10_000)
            page.wait_for_function(
                """() => (document.querySelector(
                    '#segmentation-tool [data-role="status"]'
                )?.textContent || '').startsWith('已生成独立 3D 图层')""",
                timeout=args.timeout_ms,
            )
            separate_status = tool.locator('[data-role="status"]').inner_text().strip()
            page.wait_for_function(
                "count => document.querySelectorAll('.splat-list .splat-item').length === count",
                arg=initial_splat_items + 1,
                timeout=args.timeout_ms,
            )
            separated_splat_items = page.locator('.splat-list .splat-item').count()
            separated = artifact_dir / "06_separated.png"
            page.screenshot(path=str(separated))
            result["screenshots"].append(str(separated))

            tool.locator('[data-action="cancel"]').click(timeout=10_000)
            page.locator("#segmentation-tool.active").wait_for(
                state="hidden", timeout=10_000
            )
            for _ in range(4):
                page.keyboard.press("Control+z")
                page.wait_for_timeout(500)
            undo_selection_zero = wait_for_undo_restored(
                page, initial_splat_items, timeout_ms=15_000
            )
            after_undo_selected = selected_count(page)
            after_undo_splat_items = page.locator(
                '.splat-list .splat-item'
            ).count()
            after_undo = artifact_dir / "07_cancelled_undo.png"
            page.screenshot(path=str(after_undo))
            result["screenshots"].append(str(after_undo))

            undo_or_no_pageerror = undo_selection_zero or not page_errors
            if not undo_or_no_pageerror:
                raise SmokeFailure(
                    "selection did not return to zero and pageerror was emitted"
                )
            if mock["predict_requests"] != 1:
                raise SmokeFailure(
                    f"expected one semantic predict request, got {mock['predict_requests']}"
                )
            if len({url.split("?")[0] for url in mask_requests}) != 3:
                raise SmokeFailure("browser did not request all 3 mocked mask URLs")

            result.update(
                {
                    "passed": True,
                    "initial_selected": initial_selected,
                    "initial_splat_items": initial_splat_items,
                    "projected_status": status,
                    "projected_count": projected_count,
                    "projected_selected": projected_selected,
                    "expanded_status": expanded_status,
                    "expanded_seed_count": expanded_seed_count,
                "expanded_added_count": expanded_added_count,
                "refined_status": refined_status,
                    "separate_status": separate_status,
                    "separated_splat_items": separated_splat_items,
                    "after_undo_selected": after_undo_selected,
                    "after_undo_splat_items": after_undo_splat_items,
                    "undo_selection_zero": undo_selection_zero,
                    "undo_or_no_pageerror": undo_or_no_pageerror,
                }
            )
            browser.close()
            browser = None
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        if page is not None:
            try:
                failure = artifact_dir / "99_failure.png"
                page.screenshot(path=str(failure), timeout=10_000)
                result["screenshots"].append(str(failure))
            except Exception as screenshot_error:
                result["screenshot_error"] = str(screenshot_error)
    finally:
        if browser is not None:
            try:
                browser.close()
            except Exception as close_error:
                result["browser_close_error"] = str(close_error)
        result.update(
            {
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "mock": {key: value for key, value in mock.items() if key != "mask_png"},
                "mask_request_count": len(mask_requests),
                "mask_request_urls": sorted(set(mask_requests)),
                "route_errors": route_errors,
                "page_errors": page_errors,
                "console_errors": console_errors,
            }
        )
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    sys.exit(main())
