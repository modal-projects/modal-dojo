from __future__ import annotations

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import modal
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
STUB = '<meta http-equiv="refresh" content="0;url=/guides/start/model/">'
PAGE = "<html><title>Start</title></html>"
TARGET = "/guides/start/model/"


@pytest.fixture
def docs_client(tmp_path, monkeypatch):
    dist, _ = _site(tmp_path)
    (dist / "index.html").write_text(PAGE)
    monkeypatch.syspath_prepend(str(ROOT / "docs-next"))
    monkeypatch.setattr(modal, "is_local", lambda: False)
    docs = _load("docs_next_app", ROOT / "docs-next" / "docs_next_app.py")
    monkeypatch.setattr(docs, "REMOTE_DIST", str(dist))
    with TestClient(docs.serve.local(), follow_redirects=False) as client:
        yield client


@pytest.mark.parametrize("method", ["GET", "HEAD"])
@pytest.mark.parametrize(
    "path",
    [
        "/",
        "/guides",
        "/guides/start/model/",
        "/missing?x=1&x=2&next=%2Fguides",
        "/files/a%2Fb%20c?value=%23hash",
    ],
)
def test_legacy_domain_redirect_preserves_url(docs_client, method, path):
    response = docs_client.request(method, f"https://gym.modal.dev{path}")
    assert response.status_code == 301
    assert response.headers["location"] == f"https://dojo.modal.dev{path}"


def test_legacy_domain_redirect_without_raw_path(tmp_path, monkeypatch):
    dist, _ = _site(tmp_path)
    monkeypatch.syspath_prepend(str(ROOT / "docs-next"))
    monkeypatch.setattr(modal, "is_local", lambda: False)
    docs = _load("docs_next_app", ROOT / "docs-next" / "docs_next_app.py")
    monkeypatch.setattr(docs, "REMOTE_DIST", str(dist))
    inner = docs.serve.local()

    async def without_raw_path(scope, receive, send):
        scope.pop("raw_path", None)
        await inner(scope, receive, send)

    with TestClient(without_raw_path, follow_redirects=False) as client:
        response = client.head("https://gym.modal.dev/docs/a%20b?x=1")
    assert response.status_code == 301
    assert response.headers["location"] == "https://dojo.modal.dev/docs/a%20b?x=1"


@pytest.mark.parametrize("host", ["dojo.modal.dev", "preview.modal.run"])
def test_other_domains_serve_docs_and_keep_path_redirects(docs_client, host):
    response = docs_client.get(f"https://{host}/")
    assert response.status_code == 200
    assert response.text == PAGE
    response = docs_client.get(f"https://{host}/guides?source=test")
    assert response.status_code == 302
    assert response.headers["location"] == f"{TARGET}?source=test"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _site(tmp_path: Path) -> tuple[Path, Path]:
    dist = tmp_path / "dist"
    (dist / "guides").mkdir(parents=True)
    (dist / "guides" / "index.html").write_text(STUB)
    page_dir = dist / "guides" / "start" / "model"
    page_dir.mkdir(parents=True)
    (page_dir / "index.html").write_text(PAGE)
    tarball = tmp_path / "docs.tar.gz"
    subprocess.check_call(["tar", "-czf", str(tarball), "-C", str(dist), "."])
    return dist, tarball


def test_refresh_map_and_nginx_slash_locations(tmp_path: Path) -> None:
    dist, tarball = _site(tmp_path)
    astro = _load("astro_redirects", ROOT / "docs-next" / "astro_redirects.py")
    expected = {"/guides": TARGET}
    assert astro.refresh_map(dist) == expected
    assert astro.refresh_map_from_tarball(tarball) == expected

    previews = _load(
        "frontend_previews",
        ROOT / "scripts" / "previews" / "frontend_previews.py",
    )
    snippet = previews._docs_nginx_refresh_inc(expected)
    assert f"location = /guides {{ return 302 {TARGET}$is_args$args; }}" in snippet
    assert f"location = /guides/ {{ return 302 {TARGET}$is_args$args; }}" in snippet

    page_move = {"/guides/tools/dashboard": "/guides/dashboard"}
    move_snippet = previews._docs_nginx_refresh_inc(page_move)
    assert (
        "location = /guides/tools/dashboard { return 301 /guides/dashboard$is_args$args; }"
        in move_snippet
    )
    for section_home in (
        "/guides",
        "/reference",
        "/tutorials",
        "/tutorials/agent",
        "/tutorials/rl",
    ):
        assert astro.redirect_status(section_home) == 302

    for moved_page in (
        "/guides/tools/dashboard",
        "/guides/tools/agent-driven-training",
        "/guides/agent-driven-training",
        "/reference/core/modelconfig",
        "/tutorials/rl/000_rl_basics",
    ):
        assert astro.redirect_status(moved_page) == 301


def test_old_agent_guide_slugs_redirect_to_agent() -> None:
    config = (ROOT / "docs-next" / "astro.config.mjs").read_text()
    assert "'/guides/tools/agent-driven-training': '/guides/agent'" in config
    assert "'/guides/agent-driven-training': '/guides/agent'" in config


def test_old_reference_slugs_redirect_to_flattened_pages() -> None:
    config = re.sub(
        r"\s+",
        " ",
        (ROOT / "docs-next" / "astro.config.mjs").read_text(),
    )
    assert "'/reference': '/reference/sdk'" in config
    assert "'/reference/core/modelconfig': '/reference/modelconfig'" in config
    assert "'/reference/core/trackioconfig': '/reference/trackioconfig'" in config
    assert (
        "'/reference/training/qwen3_5_4b_miles_recipe': "
        "'/reference/qwen3_5_4b_miles_recipe'" in config
    )
    assert "'/reference/models/parsedresponse': '/reference/parsedresponse'" in config
    assert (
        "'/reference/models/parse_qwen3_response': "
        "'/reference/modelconfig#parse_response'" in config
    )
    assert (
        "'/reference/parse_qwen3_response': '/reference/modelconfig#parse_response'"
        in config
    )
