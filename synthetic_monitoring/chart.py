import io
import math
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from modal_dojo.common.step_timing import Substep

if TYPE_CHECKING:
    from scripts.validate_model_configs import ValidationResult

LOOKBACK_S = 90 * 24 * 3600

_STEP_SUFFIX = re.compile(r" \(step \d+\)$")
_LANE_ORDER = {None: 0, "actor": 1, "rollout": 2, "critic": 3}
_SUBSTEP_ORDER = {member.value: i for i, member in enumerate(Substep)}

_GRAY_00 = "#181818"
_GRAY_10 = "#232323"
_GRAY_20 = "#2f2f2f"
_GRAY_30 = "#464646"
_GRAY_50 = "#a3a3a3"
_GRAY_70 = "#d1d1d1"
_DATAVIZ_GREEN_2 = "#6ac345"
_DATAVIZ_ORANGE_2 = "#ffa556"
_DATAVIZ_BLUE_2 = "#85aef8"
_DATAVIZ_PURPLE_2 = "#c29bfc"
_LANE_COLORS = {
    None: _DATAVIZ_GREEN_2,
    "actor": _DATAVIZ_BLUE_2,
    "rollout": _DATAVIZ_ORANGE_2,
    "critic": _DATAVIZ_PURPLE_2,
}


@dataclass(frozen=True)
class RunPoint:
    ts: float
    timings: dict[str, float]
    training_run_id: str
    total_duration_s: float
    status: str = "success"
    modal_app_url: str | None = None

    @classmethod
    def from_validation_result(
        cls,
        result: "ValidationResult",
        *,
        ts: float | None = None,
        modal_app_url: str | None = None,
    ) -> "RunPoint":
        timings: dict[str, float] = {}
        step_times = result.step_times or {}
        substep_times = result.substep_times or {}
        for step_key in sorted(step_times, key=int):
            ordered_substeps = sorted(
                substep_times[step_key].items(), key=lambda item: item[1]["start"]
            )
            for name, entry in ordered_substeps:
                timings[f"{name} (step {step_key})"] = (
                    entry["wall_duration_s"]
                    if entry["concurrent"]
                    else entry["duration_s"]
                )
            timings[f"Step {step_key}"] = step_times[step_key]["duration_s"]
        return cls(
            ts=time.time() if ts is None else ts,
            timings=timings,
            training_run_id=result.training_run_id,
            total_duration_s=result.total_duration_s,
            status="success" if result.succeeded else "failed",
            modal_app_url=modal_app_url,
        )


def _panel_title(label: str) -> str:
    name = _STEP_SUFFIX.sub("", label).replace("_", " ")
    return name[:1].upper() + name[1:]


def _lane(title: str) -> str | None:
    match = re.search(r"\((actor|rollout|critic)\)", title)
    return match.group(1) if match else None


def _panel_sort_key(raw: str) -> tuple[int, int, str]:
    index = _SUBSTEP_ORDER.get(raw.split(" (", 1)[0])
    if index is None:
        return (len(_SUBSTEP_ORDER), 0, raw)
    return (index, _LANE_ORDER[_lane(raw)], "")


def _substep_series(history: list[RunPoint]) -> dict[str, list[float | None]]:
    names: list[str] = []
    seen: set[str] = set()
    for point in history:
        for label in point.timings:
            if label.startswith("Step "):
                continue
            raw = _STEP_SUFFIX.sub("", label)
            if raw not in seen:
                seen.add(raw)
                names.append(raw)
    names.sort(key=_panel_sort_key)
    names = [_panel_title(raw) for raw in names]
    series: dict[str, list[float | None]] = {name: [] for name in names}
    for point in history:
        for name in names:
            total = 0.0
            found = False
            for label, value in point.timings.items():
                if label.startswith("Step "):
                    continue
                if _panel_title(label) == name:
                    found = True
                    total += float(value)
            series[name].append(total if found else None)
    return series


def _style_history_axes(ax) -> None:
    ax.set_facecolor(_GRAY_10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=_GRAY_20, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.tick_params(colors=_GRAY_50, length=3, width=0.8)
    ax.xaxis.label.set_color(_GRAY_50)
    ax.yaxis.label.set_color(_GRAY_50)
    ax.title.set_color(_GRAY_70)
    for spine in ax.spines.values():
        spine.set_color(_GRAY_30)
        spine.set_linewidth(0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_timing_history_chart(
    history: list[RunPoint],
    *,
    model_name: str,
) -> bytes:
    from matplotlib import pyplot

    cutoff = time.time() - LOOKBACK_S
    points = [p for p in history if p.status == "success" and p.ts >= cutoff]
    series = _substep_series(points)
    names = list(series)
    if not names:
        return b""

    tick_labels = [
        datetime.fromtimestamp(p.ts, tz=timezone.utc).strftime("%m-%d\n%H:%M")
        for p in points
    ]
    xs = list(range(len(points)))
    n_cols = max(1, math.ceil(math.sqrt(len(names))))
    n_rows = math.ceil(len(names) / n_cols)
    fig, axes = pyplot.subplots(
        n_rows,
        n_cols,
        figsize=(2 + 4 * n_cols, 2 + 4.2 * n_rows),
        squeeze=False,
        sharey=False,
        layout="constrained",
    )
    fig.patch.set_facecolor(_GRAY_00)
    fig.suptitle(
        f"{model_name} (n={len(points)} runs)",
        color=_GRAY_70,
        fontsize=13,
        fontweight="medium",
    )

    for i, name in enumerate(names):
        ax = axes[i // n_cols][i % n_cols]
        for x, y in zip(xs, series[name], strict=True):
            if y is None:
                continue
            ax.bar(x, y, color=_LANE_COLORS[_lane(name)], width=0.72, zorder=2)
        ax.set_title(name, fontsize=10, pad=10)
        ax.set_ylabel("seconds")
        ax.set_xticks(xs)
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=0)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        _style_history_axes(ax)
        if i // n_cols == n_rows - 1:
            ax.set_xlabel("time (UTC)")

    for j in range(len(names), n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.get_layout_engine().set(h_pad=0.18, hspace=0.18)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=_GRAY_00)
    pyplot.close(fig)
    return buf.getvalue()
