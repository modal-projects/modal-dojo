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


def test_signature_split_across_chunks_is_seen_whole() -> None:
    collector = FailureExcerpt()
    for chunk in [
        "some log\nTraceback (most recent call la",
        'st):\n  File "t.py", line 1\ntorch.OutOfMemoryEr',
        "ror: CUDA out of memory.\nrest\n",
    ]:
        collector.feed_chunk(chunk)
    assert collector.result() == "torch.OutOfMemoryError: CUDA out of memory."


def test_unterminated_tail_is_flushed_at_result() -> None:
    collector = FailureExcerpt()
    collector.feed_chunk("log\nRuntimeError: partial tail")
    assert collector.result() == "RuntimeError: partial tail"


def test_interleaved_line_inside_traceback() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "a.py", line 1, in <module>',
        "[rank 7] step 3 done",
        "megatron.core.CustomError: bespoke failure",
    ]
    assert (
        extract_failure_excerpt(lines) == "megatron.core.CustomError: bespoke failure"
    )


def test_bare_exception_line_closes_traceback() -> None:
    lines = [
        "Traceback (most recent call last):",
        '  File "a.py", line 1',
        "KeyboardInterrupt",
        "worker cleanup",
    ]
    assert extract_failure_excerpt(lines) == "KeyboardInterrupt"


def test_log_prefixed_traceback_lines() -> None:
    lines = [
        "(TrainActor pid=1) Traceback (most recent call last):",
        '(TrainActor pid=1)   File "a.py", line 1',
        "(TrainActor pid=1) megatron.core.CustomError: bespoke",
    ]
    excerpt = extract_failure_excerpt(lines)
    assert excerpt is not None
    assert "CustomError" in excerpt


def test_duplicate_signatures_are_deduplicated() -> None:
    lines = [
        "RuntimeError: same failure",
        "RuntimeError: same failure",
        "ValueError: other",
    ]
    assert (
        extract_failure_excerpt(lines)
        == "RuntimeError: same failure\nValueError: other"
    )


def test_newline_free_chunks_bound_the_tail_buffer() -> None:
    collector = FailureExcerpt()
    for _ in range(10):
        collector.feed_chunk("x" * 100_000)
    assert len(collector._buf) <= 8 * 1024 + 256


def test_carriage_return_progress_does_not_hide_exception() -> None:
    collector = FailureExcerpt()
    collector.feed_chunk("10%|\r20%||\rRuntimeError: boom\n")
    assert collector.result() == "RuntimeError: boom"


def test_long_prefixed_traceback_still_finds_cause() -> None:
    lines = ["(TrainActor pid=1) Traceback (most recent call last):"]
    lines += [
        f'(TrainActor pid=1)   File "f{i}.py", line {i}, in run' for i in range(20)
    ]
    lines.append("(TrainActor pid=1) torch.OutOfMemoryError: CUDA out of memory.")
    excerpt = extract_failure_excerpt(lines)
    assert excerpt == "torch.OutOfMemoryError: CUDA out of memory."


def test_bare_prefixed_exception_closes_traceback() -> None:
    lines = [
        "(TrainActor pid=1) Traceback (most recent call last):",
        '(TrainActor pid=1)   File "a.py", line 1',
        "(TrainActor pid=1) KeyboardInterrupt",
    ]
    assert extract_failure_excerpt(lines) == "KeyboardInterrupt"


def test_flush_separates_fragments_across_reconnect() -> None:
    collector = FailureExcerpt()
    collector.feed_chunk("RuntimeError: first")
    collector.flush()
    collector.feed_chunk("ValueError: second\n")
    assert collector.result() == "RuntimeError: first\nValueError: second"


def test_nccl_tag_is_not_stripped_as_prefix() -> None:
    lines = ["[NCCL] error: unhandled system error"]
    excerpt = extract_failure_excerpt(lines)
    assert excerpt is not None
    assert "NCCL" in excerpt


def test_oversized_line_head_still_classifies() -> None:
    collector = FailureExcerpt()
    collector.feed_chunk("RuntimeError: " + "x" * 100_000 + "\n")
    excerpt = collector.result()
    assert excerpt is not None
    assert "RuntimeError" in excerpt


def test_signature_split_at_tail_boundary_reassembles() -> None:
    collector = FailureExcerpt()
    collector.feed_chunk("-" * (8 * 1024 + 256) + "Runtime")
    collector.feed_chunk("Error: boom\n")
    excerpt = collector.result()
    assert excerpt is not None
    assert "RuntimeError" in excerpt


def test_tail_overlap_is_not_reclassified() -> None:
    collector = FailureExcerpt()
    collector.feed_chunk("a" * 300 + "RuntimeError: boom" + "b" * 8192)
    collector.feed_chunk("c" * 9000)
    collector.feed_chunk("d" * 9000)
    excerpt = collector.result()
    assert excerpt is not None
    assert excerpt.count("RuntimeError") == 1
