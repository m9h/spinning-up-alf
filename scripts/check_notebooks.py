"""Validate or execute notebooks without overwriting their checked-in outputs."""

import argparse
import json
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per cell")
    parser.add_argument("--output-dir", type=Path, default=Path("notebook-results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    paths = args.notebooks or sorted((root / "notebooks").glob("*.ipynb"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in paths:
        started = time.monotonic()
        notebook = None
        result = {"notebook": path.name, "status": "passed"}
        try:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            if args.execute:
                client = NotebookClient(
                    notebook,
                    timeout=args.timeout,
                    kernel_name="python3",
                    resources={"metadata": {"path": str(path.resolve().parent)}},
                )
                client.execute()
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            if args.execute and notebook is not None:
                nbformat.write(notebook, args.output_dir / path.name)
        result["seconds"] = round(time.monotonic() - started, 2)
        results.append(result)
        print(f"{result['status'].upper()}: {path.name} ({result['seconds']}s)", flush=True)
        if "error" in result:
            print(result["error"], flush=True)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    return int(any(result["status"] == "failed" for result in results))


if __name__ == "__main__":
    raise SystemExit(main())
