from __future__ import annotations

import importlib.util
import inspect
import sys
import tempfile
import traceback
from pathlib import Path


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_test(func) -> tuple[bool, str]:
    kwargs = {}
    temp_dir = None
    if "tmp_path" in inspect.signature(func).parameters:
        temp_dir = tempfile.TemporaryDirectory()
        kwargs["tmp_path"] = Path(temp_dir.name)
    try:
        func(**kwargs)
        return True, ""
    except Exception:
        return False, traceback.format_exc()
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def main() -> int:
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
    test_root = Path(args[0]) if args else Path("tests")
    files = sorted(test_root.rglob("test_*.py"))
    total = 0
    failed = 0
    for path in files:
        module = _load_module(path)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            obj = getattr(module, name)
            if not callable(obj):
                continue
            total += 1
            ok, error = _run_test(obj)
            if ok:
                sys.stdout.write(".")
            else:
                failed += 1
                sys.stdout.write("F")
                sys.stdout.flush()
                sys.stderr.write(f"\nFAILED {path}:{name}\n{error}\n")
            sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.write(f"{total - failed} passed, {failed} failed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
