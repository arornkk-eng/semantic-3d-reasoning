"""PLY 流式写入测试（阶段 2 护栏）。

锁定两个修复点：
1. 上传不再 `await file.read()` 整体读入内存（避免大文件 OOM）；
2. 超过上限立即中止并清理半截文件。

用 asyncio.run 驱动协程，无需额外引入 pytest-asyncio 依赖。
"""

import asyncio

import pytest
from fastapi import HTTPException

from backend.api import ply_viewer


class FakeUpload:
    """模拟分块读取的 UploadFile。"""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    async def read(self, n=-1):
        if self._chunks:
            return self._chunks.pop(0)
        return b""


def test_stream_to_disk_basic(tmp_path):
    f = FakeUpload([b"hello ply content"])
    dest = tmp_path / "a.ply"
    size = asyncio.run(ply_viewer.stream_to_disk(f, dest, max_bytes=10 * 1024 * 1024))
    assert size == len(b"hello ply content")
    assert dest.read_bytes() == b"hello ply content"


def test_stream_to_disk_enforces_size(tmp_path):
    # 5 个 1KB 分块，上限 2KB：第 3 个分块写入前即触发超限
    f = FakeUpload([b"x" * 1024 for _ in range(5)])
    dest = tmp_path / "big.ply"
    with pytest.raises(HTTPException):
        asyncio.run(ply_viewer.stream_to_disk(f, dest, max_bytes=2 * 1024))
    assert not dest.exists()  # 超限应清理半截文件
