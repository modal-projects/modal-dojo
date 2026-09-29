"""Modal app serving the Starlight documentation site.

Serves pre-built static files. Run the build locally before deploying:

    cd docs-next && npm install && npm run build

Deploy:

    uv run modal deploy docs-next/docs_next_app.py

Local development:

    cd docs-next && npm run dev
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import modal

from astro_redirects import redirect_status, refresh_map

DOCS_DIR = Path(__file__).resolve().parent
DIST_DIR = DOCS_DIR / "dist"
REMOTE_DIST = "/assets/dist"

if not modal.is_local():
    pass
elif not DIST_DIR.exists() or not any(DIST_DIR.iterdir()):
    import subprocess
    import sys

    REPO_ROOT = DOCS_DIR.parent
    subprocess.check_call(
        [sys.executable, "scripts/generate_all.py"], cwd=str(REPO_ROOT)
    )


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]==0.141.1", "starlette==1.6.0")
    .add_local_dir(DIST_DIR, remote_path=REMOTE_DIST, copy=True)
    .add_local_python_source("astro_redirects", copy=True)
)

app = modal.App("training-gym-docs", image=image)

CANONICAL_HOST = "dojo.modal.dev"
LEGACY_HOST = "gym.modal.dev"


def cache_control_value(path: str, content_type: str) -> str | None:
    if path.startswith("/_astro/"):
        return "public, max-age=31536000, immutable"
    if path.startswith("/pagefind/"):
        if path.endswith((".pf_index", ".pf_fragment", ".pf_meta")):
            return "public, max-age=31536000, immutable"
        return "public, max-age=3600, stale-while-revalidate=86400"
    if path.endswith(".html") or "text/html" in content_type:
        return "public, max-age=3600, stale-while-revalidate=86400"
    return None


@app.function(min_containers=1)
@modal.concurrent(max_inputs=100)
@modal.asgi_app(custom_domains=[CANONICAL_HOST, LEGACY_HOST])
def serve():
    from fastapi import FastAPI, Request, Response
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import RedirectResponse
    from fastapi.staticfiles import StaticFiles

    web = FastAPI()
    web.add_middleware(GZipMiddleware, minimum_size=500)
    dist = Path(REMOTE_DIST)
    redirects = refresh_map(dist)

    @web.middleware("http")
    async def directory_index_without_slash(request: Request, call_next):
        path = request.url.path
        if path != "/" and path.endswith("/"):
            target = path.rstrip("/") or "/"
            query = request.url.query
            location = f"{target}?{query}" if query else target
            return RedirectResponse(url=location, status_code=301)
        if path != "/" and (dist / path.lstrip("/") / "index.html").is_file():
            request.scope["path"] = f"{path}/"
        return await call_next(request)

    @web.middleware("http")
    async def cache_control(request: Request, call_next):
        response: Response = await call_next(request)
        value = cache_control_value(
            request.url.path, response.headers.get("content-type", "")
        )
        if value and response.status_code in (200, 206):
            response.headers["Cache-Control"] = value
        return response

    @web.middleware("http")
    async def astro_refresh_redirects(request: Request, call_next):
        if request.method not in ("GET", "HEAD"):
            return await call_next(request)
        path = request.url.path.rstrip("/") or "/"
        target = redirects.get(path)
        if target is None:
            return await call_next(request)
        query = request.url.query
        location = f"{target}?{query}" if query else target
        return RedirectResponse(url=location, status_code=redirect_status(path))

    # Registered last so legacy requests redirect before any path rewrites.
    @web.middleware("http")
    async def legacy_host_redirect(request: Request, call_next):
        if request.url.hostname != LEGACY_HOST:
            return await call_next(request)
        raw_path = request.scope.get("raw_path")
        path = raw_path.decode("ascii") if raw_path else quote(request.url.path)
        query = request.url.query
        location = f"https://{CANONICAL_HOST}{path}"
        if query:
            location += f"?{query}"
        return RedirectResponse(url=location, status_code=301)

    web.mount("/", StaticFiles(directory=REMOTE_DIST, html=True), name="static")

    return web
