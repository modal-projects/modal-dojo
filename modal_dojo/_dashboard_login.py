"""Dashboard login page and session cookies.

The cookie is an expiry plus an HMAC keyed on the password, so any container
can check it and changing the password signs everyone out.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import html
import time

LOGIN_PATH = "/login"
LOGOUT_PATH = "/logout"
SESSION_COOKIE = "modal_dojo_session"
SESSION_TTL_SECONDS = 30 * 24 * 60 * 60
LOGIN_LINK_TTL_SECONDS = 60
FAILED_LOGIN_DELAY_SECONDS = 1.0
# Tells the frontend a 401 means "log in again".
LOGIN_REQUIRED_HEADER = "X-Modal-Dojo-Login"
LOGIN_PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
}
LOGIN_PAGE_PATHS = frozenset(
    {LOGIN_PATH, LOGOUT_PATH, "/favicon.svg", "/apple-touch-icon.png"}
)

_SESSION_CONTEXT = b"modal-dojo-session:v1:"
_LOGIN_LINK_CONTEXT = b"modal-dojo-login-link:v1:"


def password_matches(supplied: str, password: str) -> bool:
    return bool(password) and hmac.compare_digest(
        supplied.encode("utf-8"), password.encode("utf-8")
    )


def basic_auth_ok(authorization: str | None, password: str) -> bool:
    scheme, _, encoded = (authorization or "").partition(" ")
    if scheme.lower() != "basic" or not encoded:
        return False
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return False
    _user, _, supplied = decoded.partition(":")
    return password_matches(supplied, password)


def bearer_token_ok(authorization: str | None, token: str) -> bool:
    scheme, _, supplied = (authorization or "").partition(" ")
    return scheme.lower() == "bearer" and password_matches(supplied.strip(), token)


def _signature(expires_at: int, password: str, context: bytes) -> str:
    return hmac.new(
        password.encode("utf-8"),
        context + str(expires_at).encode(),
        hashlib.sha256,
    ).hexdigest()


def _make_token(password: str, ttl: int, context: bytes, now: float | None) -> str:
    expires_at = int(now if now is not None else time.time()) + ttl
    return f"{expires_at}.{_signature(expires_at, password, context)}"


def _token_ok(
    token: str | None, password: str, context: bytes, now: float | None
) -> bool:
    if not token or not password:
        return False
    expires_raw, _, signature = token.partition(".")
    if not expires_raw.isdigit():
        return False
    expires_at = int(expires_raw)
    if expires_at < (now if now is not None else time.time()):
        return False
    return hmac.compare_digest(
        signature.encode("utf-8"),
        _signature(expires_at, password, context).encode(),
    )


def make_session_token(password: str, *, now: float | None = None) -> str:
    return _make_token(password, SESSION_TTL_SECONDS, _SESSION_CONTEXT, now)


def session_token_ok(
    token: str | None, password: str, *, now: float | None = None
) -> bool:
    return _token_ok(token, password, _SESSION_CONTEXT, now)


def make_login_link_token(password: str, *, now: float | None = None) -> str:
    return _make_token(password, LOGIN_LINK_TTL_SECONDS, _LOGIN_LINK_CONTEXT, now)


def login_link_token_ok(
    token: str | None, password: str, *, now: float | None = None
) -> bool:
    return _token_ok(token, password, _LOGIN_LINK_CONTEXT, now)


def safe_next_path(next_path: str | None) -> str:
    if (
        not next_path
        or not next_path.startswith("/")
        or next_path.startswith("//")
        or "\\" in next_path
        or next_path.startswith(LOGIN_PATH)
    ):
        return "/"
    return next_path


def render_login_page(*, next_path: str = "/", error: bool = False) -> str:
    error_html = (
        '<p class="error" role="alert">Incorrect password. Try again.</p>'
        if error
        else ""
    )
    return (
        _LOGIN_TEMPLATE.replace("__ERROR__", error_html)
        .replace("__INVALID__", ' aria-invalid="true"' if error else "")
        .replace("__NEXT__", html.escape(next_path))
    )


# Styled after modal.com/login.
_LOGIN_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<meta name="robots" content="noindex">
<title>Log in | Modal Dojo</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<style>
  @font-face {
    font-family: "Inter Variable";
    src: url("https://modal-cdn.com/fonts/inter-variable.woff2") format("woff2");
    font-weight: 100 900;
    font-display: swap;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; }
  body {
    background: #060805;
    color: #d1d1d1;
    font: 16px/1.5 "Inter Variable", ui-sans-serif, system-ui, sans-serif;
    -webkit-font-smoothing: antialiased;
    font-feature-settings: "cv11" on;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 24px 16px;
  }
  main { width: 100%; max-width: 360px; text-align: center; }
  .logo { display: block; width: 120px; height: 120px; margin: -28px auto -16px; }
  h1 { color: #e3e5e1; font-size: 32px; font-weight: 500; letter-spacing: -0.01em; line-height: 1.25; }
  .subtitle { color: #8b8b8b; font-size: 15px; margin: 8px 0 32px; }
  form { display: flex; flex-direction: column; gap: 16px; }
  input[type="password"] {
    width: 100%;
    height: 44px;
    padding: 0 20px;
    border-radius: 48px;
    border: 1px solid #464646;
    background: transparent;
    color: #e3e5e1;
    font: inherit;
    outline: none;
    transition: border-color 120ms;
  }
  input[type="password"]::placeholder { color: #747474; }
  input[type="password"]:focus { border-color: #7fee64; }
  input[type="password"][aria-invalid="true"] { border-color: #f87171; }
  button {
    height: 44px;
    border: 0;
    border-radius: 48px;
    background: #7fee64;
    color: #0b0d0a;
    font: inherit;
    font-weight: 500;
    cursor: pointer;
    transition: background 120ms;
  }
  button:hover { background: #a3f58d; }
  button:focus-visible { outline: 2px solid #bff9b4; outline-offset: 2px; }
  .error { color: #f87171; font-size: 14px; margin-top: -4px; text-align: left; padding-left: 20px; }
  hr { border: 0; border-top: 1px solid #272727; margin: 40px -24px 24px; }
  .hint { color: #8b8b8b; font-size: 14px; }
  code {
    color: #bababa;
    font: 13px "Fira Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
    background: #1c1c1c;
    border-radius: 6px;
    padding: 2px 6px;
  }
  .visually-hidden { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
</style>
</head>
<body>
<main>
  <img class="logo" src="/favicon.svg" alt="Modal">
  <h1>Log in to Modal Dojo</h1>
  <p class="subtitle">This dashboard is password protected.</p>
  <form method="post" action="/login">
    <input type="hidden" name="next" value="__NEXT__">
    <input class="visually-hidden" type="text" name="username" value="modal-dojo" autocomplete="username" tabindex="-1" aria-hidden="true">
    <label class="visually-hidden" for="password">Dashboard password</label>
    <input id="password" type="password" name="password" placeholder="Dashboard password" autocomplete="current-password" required autofocus__INVALID__>
    __ERROR__
    <button type="submit">Continue</button>
  </form>
  <hr>
  <p class="hint">Forgot it? Run <code>modal-dojo set-password</code></p>
</main>
</body>
</html>
"""
