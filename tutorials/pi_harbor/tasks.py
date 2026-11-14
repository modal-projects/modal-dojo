import random
import re
import shutil
from fractions import Fraction
from pathlib import Path

TEMPLATE = Path(__file__).parent / "task"
SIZES = {"train": 512, "eval": 64}
PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}

TASK_TOML = """\
version = "1.0"

[agent]
timeout_sec = 900.0

[verifier]
timeout_sec = 60.0

[environment]
cpus = 1
memory_mb = 1024
"""


def tree(value: int, numbers: int, rng: random.Random):
    if numbers == 1:
        return value
    left = rng.randint(1, numbers - 1)
    divisors = [d for d in range(2, 13) if value and value % d == 0]
    op = rng.choice(["+", "-"] + ["*"] * bool(divisors) + ["/"] * (abs(value) <= 200))
    if op == "+":
        b = rng.randint(1, 99)
        a = value - b
    elif op == "-":
        b = rng.randint(1, 99)
        a = value + b
    elif op == "*":
        b = rng.choice(divisors)
        a = value // b
    else:
        b = rng.randint(2, 9)
        a = value * b
    return op, tree(a, left, rng), tree(b, numbers - left, rng)


def render(node, parent: str = "", right: bool = False) -> str:
    if isinstance(node, int):
        return str(node) if node >= 0 else f"({node})"
    op, left, right_node = node
    text = f"{render(left, op)} {op} {render(right_node, op, right=True)}"
    if parent and (
        PRECEDENCE[op] < PRECEDENCE[parent]
        or (right and PRECEDENCE[op] == PRECEDENCE[parent] and parent in "-/")
    ):
        return f"({text})"
    return text


def problem(rng: random.Random) -> tuple[str, int]:
    value = rng.randint(-999, 999)
    expression = render(tree(value, rng.randint(5, 50), rng))
    assert eval(re.sub(r"\d+", r"Fraction(\g<0>)", expression)) == value
    return expression, value


def write_tasks(root: Path):
    rng = random.Random(0)
    shutil.rmtree(root, ignore_errors=True)
    for split, size in SIZES.items():
        for n in range(size):
            name = f"{split}-{n:03d}"
            expression, value = problem(rng)
            shutil.copytree(TEMPLATE, root / name)
            (root / name / "instruction.md").write_text(f"Compute this and write the result to /app/answer.txt:\n\n{expression}\n")
            (root / name / "task.toml").write_text(TASK_TOML)
            (root / name / "tests" / "answer.txt").write_text(str(value))
