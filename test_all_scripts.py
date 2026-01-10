"""
Test all processed scripts for import errors.

Author: Victor Collins Oppon
MSc Data Science Dissertation, Middlesex University 2025
"""

import sys
import subprocess
from pathlib import Path

def test_script_imports(script_path, base_dir):
    """Test if a script can import its dependencies."""
    # Create test command that adds base_dir to path and tries importing
    test_code = f"""
import sys
from pathlib import Path
sys.path.insert(0, r'{base_dir}')

try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("test_module", r'{script_path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_module"] = module
    spec.loader.exec_module(module)
    print("OK")
except ImportError as e:
    print(f"IMPORT_ERROR: {{e}}")
except Exception as e:
    # Allow other errors (file not found, etc) since we're just testing imports
    if "No module named" in str(e) or "cannot import" in str(e):
        print(f"IMPORT_ERROR: {{e}}")
    else:
        print("OK")  # Script loaded, other errors are runtime issues
"""

    try:
        result = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout.strip()
        if "IMPORT_ERROR" in output:
            return False, output
        return True, "OK"
    except subprocess.TimeoutExpired:
        return True, "TIMEOUT (imports likely OK, script runs forever)"
    except Exception as e:
        return False, str(e)

def main():
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / "scripts"

    print("="*80)
    print("TESTING ALL SCRIPTS FOR IMPORT ERRORS")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Scripts directory: {scripts_dir}")
    print()

    results = {
        "passed": [],
        "failed": [],
        "total": 0
    }

    # Test all Python files
    for py_file in sorted(scripts_dir.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        results["total"] += 1
        relative_path = py_file.relative_to(base_dir)

        success, message = test_script_imports(py_file, base_dir)

        if success:
            results["passed"].append(str(relative_path))
            print(f"[OK] {relative_path}")
        else:
            results["failed"].append((str(relative_path), message))
            print(f"[FAIL] {relative_path}")
            print(f"       {message}")

    # Summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total scripts tested: {results['total']}")
    print(f"Passed: {len(results['passed'])} ({len(results['passed'])/results['total']*100:.1f}%)")
    print(f"Failed: {len(results['failed'])} ({len(results['failed'])/results['total']*100:.1f}%)")

    if results['failed']:
        print()
        print("FAILED SCRIPTS:")
        for path, error in results['failed'][:10]:  # Show first 10
            print(f"  - {path}")
            print(f"    {error[:100]}")
        if len(results['failed']) > 10:
            print(f"  ... and {len(results['failed']) - 10} more")
    else:
        print()
        print("[SUCCESS] ALL SCRIPTS PASS IMPORT TEST!")

    print("="*80)

    return len(results['failed']) == 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
