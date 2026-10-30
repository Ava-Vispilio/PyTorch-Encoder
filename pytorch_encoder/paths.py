"""Repository paths (outputs, fixtures, examples)."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = REPO_ROOT / "outputs"
FIXTURES = REPO_ROOT / "fixtures"
EXAMPLES = REPO_ROOT / "examples"
DEFAULT_BENCHMARK = FIXTURES / "accuracy_comparison.json"
DEFAULT_TRACE = OUTPUTS / "vm_run" / "vm_trace.jsonl"
