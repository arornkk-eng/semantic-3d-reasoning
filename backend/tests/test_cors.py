"""CORS 安全配置测试（阶段 2 护栏）。

锁定「绝不开放通配符 * 与凭证同时开启」这条安全底线：
Starlette 在 allow_origins=['*'] + allow_credentials=True 时会反射任意来源并带凭证，等同敞开。
"""

import backend.core.config as config


def test_resolve_cors_default_no_wildcard():
    """默认配置应返回显式本地来源，且不含 '*'、允许凭证。"""
    origins, allow_credentials, origin_regex = config.resolve_cors()
    assert "*" not in origins
    assert allow_credentials is True
    assert any("localhost" in o for o in origins)
    assert origin_regex is not None


def test_resolve_cors_env_override(monkeypatch):
    """环境变量 ZIPSPLAT_CORS_ORIGINS 应被解析为显式白名单。"""
    monkeypatch.setenv(
        "ZIPSPLAT_CORS_ORIGINS", "https://app.example.com, https://admin.example.com"
    )
    origins, allow_credentials, _ = config.resolve_cors()
    assert origins == ["https://app.example.com", "https://admin.example.com"]
    assert allow_credentials is True


def test_resolve_cors_wildcard_disables_credentials(monkeypatch):
    """即便有人误配 '*'，也必须关闭凭证，杜绝反射任意来源+凭证的漏洞。"""
    monkeypatch.setenv("ZIPSPLAT_CORS_ORIGINS", "*")
    origins, allow_credentials, _ = config.resolve_cors()
    assert origins == ["*"]
    assert allow_credentials is False


def test_lan_origin_regex():
    """局域网 IP 来源应被正则放行，非局域网来源不放行。"""
    import re

    rx = re.compile(config.LAN_ORIGIN_REGEX)
    assert rx.match("http://192.168.1.5:5173")
    assert rx.match("http://10.0.0.2:3000")
    assert not rx.match("http://192.168.1.5:8080")
    assert not rx.match("http://evil.example.com:5173")
