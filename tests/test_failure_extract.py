from modal_dojo.common.failure_extract import (
    FailureExcerpt,
    extract_failure_excerpt,
)


def test_extracts_exception_line_from_first_traceback() -> None:
    lines = [
        "[rank 3] starting step",
        "Traceback (most recent call last):",
        '  File "train.py", line 10, in <module>',
        "    run()",
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.8 GiB",
        "[rank 3] cleanup",
    ]
    excerpt = extract_failure_excerpt(lines)
    assert excerpt == (
        "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.8 GiB"
    )


def test_follows_chained_exceptions_to_terminal_cause() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "a.py", line 1',
        "RuntimeError: original",
        "",
        "During handling of the above exception, another exception occurred:",
        "",
        "Traceback (most recent call last):",
        '  File "b.py", line 2',
        "ValueError: the real cause",
        "[rank 0] done",
    ]
    assert extract_failure_excerpt(lines) == "ValueError: the real cause"


def test_falls_back_to_signature_when_no_traceback() -> None:
    lines = [
        "[rank 5] waiting on alltoall",
        "Watchdog caught collective operation timeout: ALLTOALL_BASE",
        "[rank 5] worker exit",
    ]
    excerpt = extract_failure_excerpt(lines)
    assert "Watchdog caught collective operation timeout" in excerpt


def test_returns_none_for_clean_logs() -> None:
    lines = ["step 1 loss=0.5", "step 2 loss=0.4"]
    assert extract_failure_excerpt(lines) is None


def test_caps_excerpt_at_max_lines() -> None:
    lines = [f"RuntimeError: failure {i}" for i in range(10)]
    excerpt = extract_failure_excerpt(lines)
    assert excerpt is not None
    assert len(excerpt.splitlines()) == 4


def test_clean_exit_code_does_not_match() -> None:
    lines = ["[rank 0] exited with code 0", "step 1 loss=0.5"]
    assert extract_failure_excerpt(lines) is None


def test_tolerates_interleaved_lines_in_chained_traceback() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "a.py", line 1',
        "RuntimeError: original",
        "",
        "During handling of the above exception, another exception occurred:",
        "",
        "[rank 7] step 3 done",
        "[rank 2] rollout progress",
        "Traceback (most recent call last):",
        '  File "b.py", line 2',
        "ValueError: the real cause",
    ]
    assert extract_failure_excerpt(lines) == "ValueError: the real cause"


def test_streaming_collector_survives_long_logs() -> None:
    # A failure early in a long log is still attributed; nothing is evicted.
    collector = FailureExcerpt()
    for line in [
        "Traceback (most recent call last):",
        '  File "train.py", line 10',
        "torch.OutOfMemoryError: CUDA out of memory.",
        "worker continues",
    ]:
        collector.feed(line)
    for i in range(10_000):
        collector.feed(f"[rank 0] rollout line {i}")
    assert collector.result() == "torch.OutOfMemoryError: CUDA out of memory."
