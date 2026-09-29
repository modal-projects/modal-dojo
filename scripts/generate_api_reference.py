from __future__ import annotations

import argparse
import dataclasses
import html
import importlib
import inspect
import json
import re
import shutil
from collections import OrderedDict
from dataclasses import fields as dataclass_fields
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    NamedTuple,
    NotRequired,
    get_origin,
    get_type_hints,
    is_typeddict,
)

import click
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from api_reference_manifest import (
    API_REFERENCE_MANIFEST,
    GROUPS,
    entry_sort_key,
)
from modal_dojo.cli import entrypoint_cli

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "docs-next" / "src" / "content" / "docs" / "reference"
SIDEBAR_PATH = ROOT / "docs-next" / "src" / "generated" / "reference-sidebar.json"
CLI_PROG = "modal-dojo"


_SPHINX_ROLE = re.compile(r":(?:class|meth|func|attr|mod|exc|data|const):`([^`]+)`")
_DEFAULT_TAIL = re.compile(r"\s*Default\s+`[^`]+`\.?\s*$")
_QUOTED_ANNOTATION = re.compile(r""": (['"])(.*?)\1""")
_QUOTED_RETURN = re.compile(r""" -> (['"])(.*?)\1""")
_MODULE_PREFIX = re.compile(r"\b(?:modal_dojo|modal)(?:\.\w+)+\.(\w+)")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_MAX_DEFAULT_CHARS = 80


class _RaiseDoc(NamedTuple):
    type: str
    description: str


class _DocSections(NamedTuple):
    description: str
    params: dict[str, str]
    attributes: dict[str, str]
    yields: str
    returns: str
    raises: tuple[_RaiseDoc, ...]
    examples: str
    see_also: str


class _MemberDoc(NamedTuple):
    name: str
    declaration: str
    sections: _DocSections
    callable: Any
    is_property: bool


def _is_dataclass(cls: type) -> bool:
    return hasattr(cls, "__dataclass_fields__")


def _is_pydantic_model(cls: type) -> bool:
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def _uses_structural_field_rendering(cls: type) -> bool:
    return _is_dataclass(cls) or is_typeddict(cls) or _is_pydantic_model(cls)


def _factory_default(factory: Any) -> Any:
    try:
        return factory()
    except Exception:
        return inspect.Parameter.empty


def _get_class_attrs(cls: type) -> dict[str, tuple[type, Any]]:
    hints = {}
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        for klass in reversed(cls.__mro__):
            hints.update(getattr(klass, "__annotations__", {}))

    attrs: dict[str, tuple[type, Any]] = {}

    if is_typeddict(cls):
        optional_keys = getattr(cls, "__optional_keys__", frozenset())
        for name, type_hint in hints.items():
            if name.startswith("_"):
                continue
            if name in optional_keys and get_origin(type_hint) is not NotRequired:
                type_hint = (
                    f"NotRequired[{type_hint}]"
                    if isinstance(type_hint, str)
                    else NotRequired[type_hint]
                )
            attrs[name] = (type_hint, inspect.Parameter.empty)
        return attrs

    if _is_dataclass(cls):
        MISSING = dataclasses.MISSING
        for f in dataclass_fields(cls):
            if f.name.startswith("_"):
                continue
            if f.default is not MISSING:
                default = f.default
            elif f.default_factory is not MISSING:
                default = _factory_default(f.default_factory)
            else:
                default = inspect.Parameter.empty
            attrs[f.name] = (hints.get(f.name, Any), default)
        return attrs

    if _is_pydantic_model(cls):
        for name, info in getattr(cls, "model_fields", {}).items():
            if name.startswith("_"):
                continue
            attrs[name] = (hints.get(name, Any), _pydantic_field_default(info))
        return attrs

    init_defaults: dict[str, Any] = {}
    init_method = getattr(cls, "__init__", None)
    if init_method:
        try:
            sig = inspect.signature(init_method)
            for pname, param in sig.parameters.items():
                if pname == "self":
                    continue
                if param.default is not param.empty:
                    init_defaults[pname] = param.default
        except (ValueError, TypeError):
            pass

    for name, type_hint in hints.items():
        if name.startswith("_"):
            continue
        default = inspect.getattr_static(cls, name, inspect.Parameter.empty)
        if isinstance(default, property):
            default = inspect.Parameter.empty
        if default is inspect.Parameter.empty and name in init_defaults:
            default = init_defaults[name]
        attrs[name] = (type_hint, default)

    return attrs


def _pydantic_field_default(info: Any) -> Any:
    if getattr(info, "is_required", lambda: False)():
        return inspect.Parameter.empty
    default = getattr(info, "default", inspect.Parameter.empty)
    if default is not inspect.Parameter.empty and default is not PydanticUndefined:
        return default
    factory = getattr(info, "default_factory", None)
    if callable(factory):
        return _factory_default(factory)
    return inspect.Parameter.empty


def _first_sentence(text: str) -> str:
    collapsed = " ".join(text.split())
    if not collapsed:
        return ""
    for i, ch in enumerate(collapsed):
        if ch not in ".!?":
            continue
        if i + 1 < len(collapsed) and collapsed[i + 1] not in " \n":
            continue
        prev = collapsed[:i]
        if ch == ".":
            next_char = collapsed[i + 1 :].lstrip()[:1]
            if next_char and prev.lower().endswith(("e.g", "i.e", "vs")):
                continue
            if next_char.islower() and prev.lower().endswith("etc"):
                continue
            if next_char.islower() and re.search(r"(?:\b[A-Za-z]\.)+[A-Za-z]$", prev):
                continue
        return collapsed[: i + 1]
    return collapsed


def _class_lede(obj: Any) -> str:
    sections = _parse_docstring_sections(getattr(obj, "__doc__", None) or "")
    return _first_sentence(sections.description)


def _rst_to_md(text: str) -> str:
    text = _SPHINX_ROLE.sub(r"`\1`", text)
    return re.sub(r"``(.*?)``", r"`\1`", text)


def _parse_docstring_sections(docstring: str) -> _DocSections:
    lines = inspect.cleandoc(docstring or "").splitlines()
    section_headers = {
        "Args:": "params",
        "Parameters:": "params",
        "Attributes:": "attributes",
        "Returns:": "returns",
        "Yields:": "yields",
        "Raises:": "raises",
        "Examples:": "examples",
        "Example:": "examples",
        "See Also:": "see_also",
        "See also:": "see_also",
    }
    ranges: dict[str, tuple[int, int]] = {}
    for index, line in enumerate(lines):
        header = section_headers.get(line.strip())
        if header is None or header in ranges:
            continue
        header_indent = len(line) - len(line.lstrip())
        end = index + 1
        in_fence = False
        while end < len(lines):
            candidate = lines[end]
            stripped = candidate.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                end += 1
                continue
            if in_fence or not stripped:
                end += 1
                continue
            indent = len(candidate) - len(candidate.lstrip())
            if indent <= header_indent:
                break
            end += 1
        ranges[header] = (index, end)

    covered = {index for start, end in ranges.values() for index in range(start, end)}
    description = _rst_to_md(
        "\n".join(line for index, line in enumerate(lines) if index not in covered)
    ).strip()

    def section_body(name: str) -> str:
        section_range = ranges.get(name)
        if section_range is None:
            return ""
        start, end = section_range
        body = lines[start + 1 : end]
        nonempty = [line for line in body if line.strip()]
        if not nonempty:
            return ""
        indent = min(len(line) - len(line.lstrip()) for line in nonempty)
        return _rst_to_md(
            "\n".join(line[indent:] if line.strip() else "" for line in body)
        ).strip()

    def parse_named_items(section_name: str) -> dict[str, str]:
        items: dict[str, str] = {}
        item_range = ranges.get(section_name)
        if item_range is None:
            return items
        start, end = item_range
        body = lines[start + 1 : end]
        nonempty = [line for line in body if line.strip()]
        if nonempty:
            item_indent = min(len(line) - len(line.lstrip()) for line in nonempty)
            current_name: str | None = None
            current_description: list[str] = []

            def flush_item() -> None:
                nonlocal current_name, current_description
                if current_name is not None:
                    items[current_name] = _DEFAULT_TAIL.sub(
                        "",
                        _rst_to_md(" ".join(current_description).strip()),
                    )
                current_name = None
                current_description = []

            for line in body:
                if not line.strip():
                    continue
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())
                if indent == item_indent and ":" in stripped:
                    lhs, detail = stripped.split(":", 1)
                    match = re.match(
                        r"^([*]{0,2}[A-Za-z_]\w*)(?:\s*\([^)]+\))?$",
                        lhs.strip(),
                    )
                    if match:
                        flush_item()
                        current_name = match.group(1).lstrip("*")
                        if detail.strip():
                            current_description.append(detail.strip())
                        continue
                if current_name is not None:
                    current_description.append(stripped)
            flush_item()
        return items

    params = parse_named_items("params")
    attributes = parse_named_items("attributes")

    raises: list[_RaiseDoc] = []
    raises_range = ranges.get("raises")
    if raises_range is not None:
        start, end = raises_range
        body = lines[start + 1 : end]
        nonempty = [line for line in body if line.strip()]
        if nonempty:
            item_indent = min(len(line) - len(line.lstrip()) for line in nonempty)
            current_type: str | None = None
            current_description: list[str] = []

            def flush_raise() -> None:
                nonlocal current_type, current_description
                if current_type is not None:
                    raises.append(
                        _RaiseDoc(
                            current_type,
                            _rst_to_md(" ".join(current_description).strip()),
                        )
                    )
                current_type = None
                current_description = []

            for line in body:
                if not line.strip():
                    continue
                stripped = line.strip()
                indent = len(line) - len(line.lstrip())
                if indent == item_indent and ":" in stripped:
                    flush_raise()
                    current_type, detail = stripped.split(":", 1)
                    current_type = current_type.strip()
                    if detail.strip():
                        current_description.append(detail.strip())
                    continue
                if current_type is not None:
                    current_description.append(stripped)
            flush_raise()

    return _DocSections(
        description=description,
        params=params,
        attributes=attributes,
        yields=section_body("yields"),
        returns=section_body("returns"),
        raises=tuple(raises),
        examples=section_body("examples"),
        see_also=section_body("see_also"),
    )


def _orders_within_group() -> dict[str, int]:
    next_order: dict[str, int] = {}
    orders: dict[str, int] = {}
    for entry in sorted(API_REFERENCE_MANIFEST, key=entry_sort_key):
        group = entry["group"]
        orders[entry["class_name"]] = next_order.get(group, 0)
        next_order[group] = next_order.get(group, 0) + 1
    return orders


def _page_heading(order: int, title: str) -> list[str]:
    return [
        "---",
        f"order: {order}",
        "---",
        "",
        f"# {title}",
        "",
    ]


def _format_type(type_hint: Any) -> str:
    if type_hint is inspect.Parameter.empty:
        return ""
    raw = (
        type_hint if isinstance(type_hint, str) else inspect.formatannotation(type_hint)
    )
    raw = raw.strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        raw = raw[1:-1]
    raw = raw.replace("typing.", "")
    raw = raw.replace("collections.abc.", "")
    raw = re.sub(r"<(?:class|enum) '([^']+)'>", r"\1", raw)
    raw = _MODULE_PREFIX.sub(r"\1", raw)
    if raw.startswith("Optional[") and raw.endswith("]"):
        raw = f"{raw[9:-1]} | None"
    return raw


def _omit_default_text(formatted: str) -> bool:
    raw = formatted.replace("`", "")
    return not raw or "\n" in raw or len(raw) > _MAX_DEFAULT_CHARS


def _format_default(val: Any) -> str:
    if val is inspect.Parameter.empty:
        return ""
    formatted: str
    if inspect.isroutine(val):
        name = getattr(val, "__name__", "")
        if name and not name.startswith("<"):
            formatted = f"`{name}`"
        else:
            formatted = f"`{val!r}`"
    elif dataclasses.is_dataclass(val) and not isinstance(val, type):
        formatted = f"`{type(val).__name__}(...)`"
    elif isinstance(val, str):
        formatted = f'`"{val}"`' if val else '`""`'
    elif isinstance(val, bool):
        formatted = f"`{val}`"
    elif isinstance(val, Enum):
        formatted = f"`{val.value}`"
    elif isinstance(val, (int, float)):
        formatted = f"`{val}`"
    elif val is None:
        formatted = "`None`"
    elif isinstance(val, list) and not val:
        formatted = "`[]`"
    elif isinstance(val, dict) and not val:
        formatted = "`{}`"
    else:
        formatted = f"`{val!r}`"
    if _omit_default_text(formatted):
        return ""
    return formatted


def _clean_signature(sig: inspect.Signature) -> str:
    text = str(sig)
    text = _QUOTED_ANNOTATION.sub(lambda m: f": {m.group(2)}", text)
    text = _QUOTED_RETURN.sub(lambda m: f" -> {m.group(2)}", text)
    text = text.replace("typing.", "")
    text = text.replace("collections.abc.", "")
    text = _MODULE_PREFIX.sub(r"\1", text)
    return re.sub(r"'([A-Z][A-Za-z0-9_]*)'", r"\1", text)


def _callable_signature(attr: Any) -> inspect.Signature | None:
    try:
        signature = inspect.signature(attr)
    except (ValueError, TypeError):
        return None
    parameters = [
        parameter
        for name, parameter in signature.parameters.items()
        if name not in {"self", "cls"}
    ]
    return signature.replace(parameters=parameters)


def _own_field_docs(
    cls: type, *, include_constructor_params: bool = True
) -> dict[str, str]:
    doc = cls.__dict__.get("__doc__") or ""
    sections = _parse_docstring_sections(doc)
    merged = dict(sections.attributes)
    if include_constructor_params:
        merged = {**sections.params, **merged}
    return merged


def _extract_field_docs_from_mro(
    cls: type, *, include_constructor_params: bool = True
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for klass in reversed(cls.__mro__):
        merged.update(
            _own_field_docs(
                klass, include_constructor_params=include_constructor_params
            )
        )
    return merged


def _inline_html(text: str) -> str:
    parts: list[str] = []
    last = 0
    for match in _INLINE_CODE.finditer(text):
        parts.append(html.escape(text[last : match.start()], quote=False))
        parts.append(f"<code>{html.escape(match.group(1), quote=False)}</code>")
        last = match.end()
    parts.append(html.escape(text[last:], quote=False))
    return "".join(parts)


def _shown_default(default: Any) -> str:
    raw = _format_default(default).replace("`", "")
    return "" if raw == "None" else raw


def _render_param_list(
    attrs: dict[str, tuple[Any, Any]],
    field_docs: dict[str, str],
    *,
    documented_only: bool = False,
) -> list[str]:
    lines: list[str] = []
    for name, (type_hint, default) in attrs.items():
        desc = field_docs.get(name.lstrip("*"), "").strip()
        if documented_only and not desc:
            continue
        type_str = _format_type(type_hint)
        header = f"<strong>{html.escape(name, quote=False)}</strong>"
        if type_str:
            header += f" <code>{html.escape(type_str, quote=False)}</code>"
        lines.append('<div class="tg-param">')
        lines.append(f"<p>{header}</p>")
        details: list[str] = []
        if desc:
            details.append(_inline_html(desc))
        default_text = _shown_default(default)
        if default_text:
            details.append(
                f'<span class="tg-param-default">Default: '
                f"{html.escape(default_text, quote=False)}</span>"
            )
        if details:
            lines.append(f"<p>{' '.join(details)}</p>")
        lines.append("</div>")
        lines.append("")
    return lines


def _named_section_lines(title: str, body: str) -> list[str]:
    if not body.strip():
        return []
    return [f"**{title}**", "", body.strip(), ""]


def _structured_section_lines(sections: _DocSections) -> list[str]:
    lines: list[str] = []
    if sections.yields:
        lines.extend(_named_section_lines("Yields", sections.yields))
    if sections.returns:
        lines.extend(_named_section_lines("Returns", sections.returns))
    if sections.raises:
        lines.extend(["**Raises**", ""])
        for exception in sections.raises:
            detail = f"- `{exception.type}`"
            if exception.description:
                detail += f": {exception.description}"
            lines.append(detail)
        lines.append("")
    if sections.examples:
        lines.append("**Usage**")
        lines.append("")
        lines.extend([sections.examples, ""])
    if sections.see_also:
        lines.append("**See Also**")
        lines.append("")
        lines.extend([sections.see_also, ""])
    return lines


def _page_doc(obj: Any) -> str:
    if inspect.isfunction(obj) or inspect.isroutine(obj):
        return inspect.getdoc(obj) or ""
    return obj.__dict__.get("__doc__") or ""


def _page_preamble(cls: type, entry: dict, order: int) -> list[str]:
    module_path = entry["module"]
    lines = [
        *_page_heading(order, entry["sidebar_label"]),
        "```python",
        f"from {module_path} import {entry['class_name']}",
        "```",
        "",
    ]

    class_sections = _parse_docstring_sections(_page_doc(cls))
    if class_sections.description:
        lines.append(class_sections.description)
        lines.append("")

    lines.extend(_structured_section_lines(class_sections))
    return lines


def _members_section(cls: type) -> list[str]:
    members = _get_members(cls)
    if not members:
        return []
    lines: list[str] = []
    for member in members:
        lines.append(f"## `{member.name}`")
        lines.append("")
        if member.declaration:
            lines.append("```python")
            lines.append(member.declaration)
            lines.append("```")
            lines.append("")
        if member.sections.description:
            lines.append(member.sections.description)
            lines.append("")
        if not member.is_property:
            params = _callable_params(member.callable)
            rendered_params = _render_param_list(
                params, member.sections.params, documented_only=True
            )
            if rendered_params:
                lines.append("**Parameters**")
                lines.append("")
                lines.extend(rendered_params)
        lines.extend(_structured_section_lines(member.sections))
    return lines


def _property_declaration(name: str, getter: Any) -> str:
    try:
        return_type = get_type_hints(getter).get("return", inspect.Parameter.empty)
    except Exception:
        try:
            return_type = inspect.signature(getter).return_annotation
        except (ValueError, TypeError):
            return_type = inspect.Parameter.empty
    type_name = _format_type(return_type)
    return f"{name}: {type_name}" if type_name else name


def _get_members(cls: type) -> list[_MemberDoc]:
    if is_typeddict(cls):
        return []

    field_names = set(_get_class_attrs(cls))
    members: list[_MemberDoc] = []
    for name in dir(cls):
        if name.startswith("_") or name in field_names:
            continue
        static = inspect.getattr_static(cls, name, None)
        is_property = isinstance(static, property)
        if is_property:
            attr = static.fget
            declaration = _property_declaration(name, attr)
        else:
            attr = getattr(cls, name, None)
        if attr is None or not callable(attr) or inspect.isclass(attr):
            continue
        if not getattr(attr, "__module__", "").startswith("modal_dojo"):
            continue
        if not is_property:
            signature = _callable_signature(attr)
            declaration = (
                f"{name}{_clean_signature(signature)}"
                if signature is not None
                else f"{name}()"
            )
        doc_src = inspect.getdoc(attr) or ""
        sections = _parse_docstring_sections(doc_src)
        members.append(_MemberDoc(name, declaration, sections, attr, is_property))
    return members


def _callable_params(attr: Any) -> dict[str, tuple[Any, Any]]:
    signature = _callable_signature(attr)
    if signature is None:
        return {}
    params: dict[str, tuple[Any, Any]] = {}
    for name, parameter in signature.parameters.items():
        prefix = ""
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
        elif parameter.kind is inspect.Parameter.VAR_KEYWORD:
            prefix = "**"
        params[f"{prefix}{name}"] = (parameter.annotation, parameter.default)
    return params


def _named_init_params(cls: type) -> dict[str, inspect.Parameter]:
    init = getattr(cls, "__init__", None)
    if not init:
        return {}
    try:
        sig = inspect.signature(init)
    except (ValueError, TypeError):
        return {}
    params: dict[str, inspect.Parameter] = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        prefix = ""
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            prefix = "*"
        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            prefix = "**"
        params[f"{prefix}{name}"] = param
    return params


def _fields_sections(cls: type, *, documented_only: bool = False) -> list[str]:
    attrs = _get_class_attrs(cls)
    if not attrs:
        return []
    field_docs = _extract_field_docs_from_mro(
        cls,
        include_constructor_params=not documented_only,
    )
    cards = _render_param_list(attrs, field_docs, documented_only=documented_only)
    if not cards:
        return []
    return ["**Attributes**", "", *cards]


def _constructor_section(cls: type, entry: dict) -> list[str]:
    init = cls.__dict__.get("__init__")
    if init is None:
        return []
    params = _named_init_params(cls)
    if not params:
        return []
    sections = _parse_docstring_sections(getattr(init, "__doc__", None) or "")
    if not sections.params and all(
        param.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD}
        for param in params.values()
    ):
        return []

    lines = [
        "## Constructor",
        "",
        "```python",
    ]
    try:
        signature = _clean_signature(inspect.signature(cls))
    except (ValueError, TypeError):
        signature = "()"
    lines.append(f"{entry['class_name']}{signature}")
    lines.append("```")
    lines.append("")
    if sections.description:
        lines.extend([sections.description, ""])
    constructor_attrs = {
        pname: (
            param.annotation if param.annotation is not param.empty else Any,
            param.default,
        )
        for pname, param in params.items()
    }
    cards = _render_param_list(constructor_attrs, sections.params, documented_only=True)
    if cards:
        lines.extend(["**Parameters**", "", *cards])
    lines.extend(_structured_section_lines(sections))
    return lines


def generate_class_page(cls: type, entry: dict, order: int) -> str:
    lines = _page_preamble(cls, entry, order)
    structural = _uses_structural_field_rendering(cls)
    if not entry.get("sidebar_excluded", False):
        lines.extend(_fields_sections(cls, documented_only=not structural))
    if not structural:
        lines.extend(_constructor_section(cls, entry))
    lines.extend(_members_section(cls))
    return "\n".join(lines)


def generate_function_page(fn: Any, entry: dict, order: int) -> str:
    lines = _page_preamble(fn, entry, order)
    signature = _callable_signature(fn)
    if signature is None:
        return "\n".join(lines)
    lines.extend(
        [
            "```python",
            f"{entry['class_name']}{_clean_signature(signature)}",
            "```",
            "",
        ]
    )
    sections = _parse_docstring_sections(_page_doc(fn))
    cards = _render_param_list(
        _callable_params(fn), sections.params, documented_only=True
    )
    if cards:
        lines.extend(["**Parameters**", "", *cards])
    return "\n".join(lines)


def generate_enum_page(cls: type, entry: dict, order: int) -> str:
    lines = _page_preamble(cls, entry, order)
    members = [f"{member.name} = {member.value!r}" for member in cls]
    if not members:
        return "\n".join(lines)
    lines.extend(["```python", *members, "```", ""])
    return "\n".join(lines)


def generate_sdk_page(obj: Any, entry: dict, order: int) -> str:
    if inspect.isfunction(obj):
        return generate_function_page(obj, entry, order)
    if inspect.isclass(obj) and issubclass(obj, Enum):
        return generate_enum_page(obj, entry, order)
    return generate_class_page(obj, entry, order)


class CliPage(NamedTuple):
    label: str
    slug: str
    panel: str
    summary: str
    command: click.Command
    children: tuple[CliPage, ...] = ()


def _sdk_link(entry: dict) -> str:
    return f"/reference/{entry['class_name'].lower()}"


def _cli_help_parts(command: click.Command) -> tuple[str, str]:
    text = inspect.cleandoc(command.help or command.short_help or "")
    if not text:
        return "", ""
    first_line, *remaining_lines = text.splitlines()
    first_line = " ".join(first_line.split())
    summary = _first_sentence(first_line)
    details = "\n".join([first_line.removeprefix(summary).strip(), *remaining_lines])
    return summary, details.strip()


def _cli_detail_lines(command: click.Command) -> list[str]:
    details = _cli_help_parts(command)[1]
    return [details, ""] if details else []


def _cli_epilog_lines(command: click.Command) -> list[str]:
    epilog = inspect.cleandoc(command.epilog or "")
    if not epilog:
        return []
    prefix = "Examples:\n"
    if epilog.startswith(prefix):
        return ["**Examples**:", "", "```bash", epilog.removeprefix(prefix), "```", ""]
    return [epilog, ""]


def _make_cli_page(
    command: click.Command,
    prefix: tuple[str, ...],
    panel: str,
    *,
    with_children: bool = False,
) -> CliPage:
    children: tuple[CliPage, ...] = ()
    if with_children and isinstance(command, click.Group) and command.commands:
        children = tuple(
            _make_cli_page(
                child,
                prefix + (name,),
                getattr(child, "panel", None) or panel,
            )
            for name, child in command.commands.items()
            if not child.hidden
        )
    return CliPage(
        label=" ".join(prefix),
        slug="-".join(prefix[1:]),
        panel=panel,
        summary=_cli_help_parts(command)[0],
        command=command,
        children=children,
    )


def collect_cli_pages() -> list[CliPage]:
    pages: list[CliPage] = []
    for name, child in entrypoint_cli.commands.items():
        if child.hidden:
            continue
        pages.append(
            _make_cli_page(
                child,
                (CLI_PROG, name),
                getattr(child, "panel", None) or "Commands",
                with_children=True,
            )
        )
    return pages


def build_reference_sidebar() -> dict[str, Any]:
    sdk_entries = sorted(
        (
            entry
            for entry in API_REFERENCE_MANIFEST
            if not entry.get("sidebar_excluded")
        ),
        key=entry_sort_key,
    )
    sdk_items = [
        {"label": entry["sidebar_label"], "link": _sdk_link(entry)}
        for entry in sdk_entries
    ]
    return {
        "sdk": sdk_items,
        "cli": [
            {"label": page.label, "link": f"/reference/cli/{page.slug}"}
            for page in collect_cli_pages()
        ],
    }


def rendered_sidebar_json(sidebar: dict[str, Any]) -> str:
    return json.dumps(sidebar, indent=2) + "\n"


def check_reference_sidebar() -> dict[str, Any]:
    sidebar = build_reference_sidebar()
    expected = rendered_sidebar_json(sidebar)
    if not SIDEBAR_PATH.is_file():
        raise SystemExit(
            f"{SIDEBAR_PATH} is missing. Run uv run scripts/generate_api_reference.py"
        )
    if SIDEBAR_PATH.read_text() != expected:
        raise SystemExit(
            f"{SIDEBAR_PATH} is out of date. Run "
            "uv run scripts/generate_api_reference.py"
        )
    return sidebar


def _cli_context(
    command: click.Command, ancestors: tuple[click.Command, ...] = ()
) -> click.Context:
    parent: click.Context | None = None
    for ancestor in ancestors:
        parent = click.Context(ancestor, parent=parent)
    return click.Context(command, parent=parent)


def _cli_value(value: Any) -> str:
    if callable(value):
        return "dynamic"
    if isinstance(value, str):
        return f"`{value}`"
    return _format_default(value)


def _cli_option_metadata(param: click.Option) -> list[str]:
    notes: list[str] = []
    if param.required:
        notes.append("required")

    if isinstance(param.show_default, str):
        notes.append(f"default {_cli_value(param.show_default)}")
    elif param.show_default and param.default is not None:
        notes.append(f"default {_cli_value(param.default)}")

    param_type = param.type
    if isinstance(param_type, (click.IntRange, click.FloatRange)):
        minimum = param_type.min
        maximum = param_type.max
        if minimum is not None and maximum is not None:
            if param_type.min_open or param_type.max_open:
                lower = "greater than" if param_type.min_open else "at least"
                upper = "less than" if param_type.max_open else "at most"
                notes.append(
                    f"range {lower} {_cli_value(minimum)} and "
                    f"{upper} {_cli_value(maximum)}"
                )
            else:
                notes.append(f"range {_cli_value(minimum)} to {_cli_value(maximum)}")
        elif minimum is not None:
            label = "greater than" if param_type.min_open else "minimum"
            notes.append(f"{label} {_cli_value(minimum)}")
        elif maximum is not None:
            label = "less than" if param_type.max_open else "maximum"
            notes.append(f"{label} {_cli_value(maximum)}")

    if isinstance(param.deprecated, str):
        reason = _first_sentence(param.deprecated).rstrip(".!?")
        notes.append(f"deprecated, {reason}")
    elif param.deprecated:
        notes.append("deprecated")
    return notes


def _cli_option_description(param: click.Option) -> str:
    description = " ".join(inspect.cleandoc(param.help or "").split())
    metadata = _cli_option_metadata(param)
    if not metadata:
        return description
    suffix = "; ".join(metadata)
    suffix = f"{suffix[:1].upper()}{suffix[1:]}."
    if not description:
        return suffix
    return f"{description.rstrip('.!?')}. {suffix}"


def _cli_param_lines(
    command: click.Command, ancestors: tuple[click.Command, ...] = ()
) -> list[str]:
    ctx = _cli_context(command, ancestors)
    options: list[tuple[str, str]] = []
    for param in command.get_params(ctx):
        if getattr(param, "hidden", False) or not isinstance(param, click.Option):
            continue
        record = param.get_help_record(ctx)
        if record is not None:
            options.append((record[0], _cli_option_description(param)))
    if not options:
        return []
    lines = ["**Options**:", ""]
    for name, description in options:
        lines.append(f"* `{name}`: {description}" if description else f"* `{name}`")
    lines.append("")
    return lines


def generate_cli_index(pages: list[CliPage]) -> str:
    lines = [
        *_page_heading(0, "CLI reference"),
        f"Commands and options for the `{CLI_PROG}` CLI.",
        "",
    ]
    by_panel: OrderedDict[str, list[CliPage]] = OrderedDict()
    for page in pages:
        by_panel.setdefault(page.panel, []).append(page)
    for panel, panel_pages in by_panel.items():
        lines += [f"## {panel}", ""]
        lines += ["| Name | Description |", "|------|-------------|"]
        for page in panel_pages:
            desc = page.summary.replace("|", "\\|")
            lines.append(f"| [`{page.label}`](/reference/cli/{page.slug}) | {desc} |")
        lines.append("")
    return "\n".join(lines)


def _cli_usage(page: CliPage) -> list[str]:
    ctx = click.Context(page.command)
    usage = f"{page.label} {' '.join(page.command.collect_usage_pieces(ctx))}".rstrip()
    return ["**Usage**:", "", "```bash", usage, "```", ""]


def _cli_anchor(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def _cli_commands(page: CliPage) -> list[str]:
    if not page.children:
        return []
    lines = ["**Commands**:", ""]
    for child in page.children:
        lines.append(
            f"* [`{child.label}`](#{_cli_anchor(child.label)}): {child.summary}"
        )
    lines.append("")
    return lines


def generate_cli_page(page: CliPage, order: int) -> str:
    lines = [*_page_heading(order, page.label)]
    if page.summary:
        lines += [page.summary, ""]
    lines += _cli_detail_lines(page.command)
    lines += _cli_usage(page)
    lines += _cli_param_lines(page.command, (entrypoint_cli,))
    lines += _cli_commands(page)
    lines += _cli_epilog_lines(page.command)
    for child in page.children:
        lines += [f"## `{child.label}`", ""]
        if child.summary:
            lines += [child.summary, ""]
        lines += _cli_detail_lines(child.command)
        lines += _cli_usage(child)
        lines += _cli_param_lines(child.command, (entrypoint_cli, page.command))
        lines += _cli_epilog_lines(child.command)
    return "\n".join(lines)


def generate_index_page(manifest: list[dict]) -> str:
    lines = [
        *_page_heading(0, "SDK reference"),
        "Types and functions in the `modal-dojo` Python SDK.",
        "",
    ]

    for group_key, group_info in sorted(GROUPS.items(), key=lambda x: x[1]["order"]):
        group_entries = sorted(
            [e for e in manifest if e["group"] == group_key],
            key=entry_sort_key,
        )
        if not group_entries:
            continue

        lines.append(f"## {group_info['label']}")
        lines.append("")
        lines.append("| Name | Description |")
        lines.append("|------|-------------|")

        for entry in group_entries:
            mod = importlib.import_module(entry["module"])
            obj = getattr(mod, entry["class_name"])
            desc = _class_lede(obj) or "-"

            link = _sdk_link(entry)
            lines.append(
                f"| [`{entry['sidebar_label']}`]({link}) | {desc.replace('|', '\\|')} |"
            )

        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate API reference pages.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for reference pages.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if the generated sidebar is out of date.",
    )
    args = parser.parse_args()
    if args.check:
        sidebar = check_reference_sidebar()
        print(
            f"Reference sidebar is up to date "
            f"({len(sidebar['sdk'])} SDK pages, {len(sidebar['cli'])} CLI commands)"
        )
        return

    output_dir = args.output_dir
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)

    SIDEBAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIDEBAR_PATH.write_text(rendered_sidebar_json(build_reference_sidebar()))

    errors = []
    generated = 0
    orders = _orders_within_group()

    for entry in API_REFERENCE_MANIFEST:
        try:
            mod = importlib.import_module(entry["module"])
            obj = getattr(mod, entry["class_name"])
        except (ImportError, AttributeError) as e:
            errors.append(f"ERROR: {entry['class_name']} ({entry['module']}): {e}")
            continue

        order = orders[entry["class_name"]]
        content = generate_sdk_page(obj, entry, order)

        group_dir = output_dir / entry["group"]
        group_dir.mkdir(parents=True, exist_ok=True)
        slug = entry["class_name"].lower()
        out_path = group_dir / f"{slug}.md"
        out_path.write_text(content)
        generated += 1
        print(
            f"  {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}"
        )

    index_content = generate_index_page(API_REFERENCE_MANIFEST)
    index_path = output_dir / "sdk.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(index_content)
    print(
        f"  {index_path.relative_to(ROOT) if index_path.is_relative_to(ROOT) else index_path}"
    )
    stale_index = output_dir / "index.md"
    if stale_index.exists():
        stale_index.unlink()

    pages = collect_cli_pages()
    cli_dir = output_dir / "cli"
    cli_dir.mkdir(parents=True, exist_ok=True)
    cli_index = cli_dir / "index.md"
    cli_index.write_text(generate_cli_index(pages))
    print(
        f"  {cli_index.relative_to(ROOT) if cli_index.is_relative_to(ROOT) else cli_index}"
    )
    for order, page in enumerate(pages, start=1):
        cli_path = cli_dir / f"{page.slug}.md"
        cli_path.write_text(generate_cli_page(page, order))
        print(
            f"  {cli_path.relative_to(ROOT) if cli_path.is_relative_to(ROOT) else cli_path}"
        )

    print(f"\nGenerated {generated} SDK pages + 1 index + {len(pages)} CLI pages")
    if errors:
        print("\nErrors:")
        for e in errors:
            print(f"  {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
