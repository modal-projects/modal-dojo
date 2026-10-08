from __future__ import annotations


def modal_app_dashboard_url(app_id: str | None) -> str:
    app_id = (app_id or "").strip()
    if not app_id:
        return ""
    return f"https://modal.com/id/{app_id}"
