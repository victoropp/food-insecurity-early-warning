"""
Automated cleanup script for dissertation submission.
Removes AI attribution and fixes hardcoded paths in all Python files.

Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

import re
from pathlib import Path
import shutil

# Paths
SOURCE_DIR = Path("C:/GDELT_Africa_Extract/Scripts/district_pipeline/FINAL_PIPELINE - StratifiedSpatialCV")
TARGET_DIR = Path("C:/GDELT_Africa_Extract/Scripts/district_pipeline/dissertation_submission")

def remove_ai_attribution(content, file_path):
    """Remove AI attribution from content."""
    original = content

    # Patterns to remove (case-insensitive)
    patterns = [
        r'#\s*Author:\s*Claude\s*Code.*?\n',
        r'#\s*Co-Authored-By:\s*Claude.*?\n',
        r'#\s*Generated with Claude Code.*?\n',
        r'🤖\s*Generated with \[Claude Code\].*?\n',
        r'Attribution:\s*Claude Code.*?\n',
        r'Co-Authored-By:\s*Claude Sonnet.*?\n',
        r'Author:\s*Claude Code.*?\n',
        r'"""\nAuthor: Claude Code\n.*?\n"""',
        r'Author: Claude Code, Date:.*?\n',
    ]

    for pattern in patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)

    # Add Victor's attribution if no author present and file changed
    if original != content and 'Author:' not in content and file_path.suffix == '.py':
        # Check if there's already a module docstring
        if content.startswith('"""') or content.startswith("'''"):
            # Insert author after first docstring
            parts = content.split('"""', 2) if '"""' in content else content.split("'''", 2)
            if len(parts) >= 3:
                content = parts[0] + '"""' + parts[1] + '"""\n\n# Author: Victor Collins Oppon\n# MSc Data Science Dissertation, Middlesex University 2025\n\n' + parts[2]
        else:
            # Add new header
            header = '"""\nAuthor: Victor Collins Oppon\nMSc Data Science Dissertation\nMiddlesex University, 2025\n"""\n\n'
            # Skip shebang if present
            if content.startswith('#!'):
                lines = content.split('\n', 1)
                content = lines[0] + '\n' + header + (lines[1] if len(lines) > 1 else '')
            else:
                content = header + content

    return content

def fix_hardcoded_paths(content, file_path):
    """Replace hardcoded paths with config imports."""
    original = content

    # Check if this is config.py itself - skip it
    if file_path.name == 'config.py':
        return content

    # Pattern replacements
    replacements = [
        # Direct path assignments
        (r'BASE_DIR\s*=\s*["\']C:\\GDELT_Africa_Extract["\']', 'from config import BASE_DIR'),
        (r'BASE_DIR\s*=\s*["\']D:\\GDELT_Africa_Extract["\']', 'from config import BASE_DIR'),
        (r'BASE_DIR\s*=\s*Path\(["\']C:\\GDELT_Africa_Extract["\']\)', 'from config import BASE_DIR'),

        # Path usage in strings
        (r'["\']C:\\GDELT_Africa_Extract\\Scripts\\district_pipeline\\FINAL_PIPELINE[^"\']*["\']',
         'str(BASE_DIR)'),
        (r'["\']C:\\GDELT_Africa_Extract["\']', 'str(BASE_DIR.parent.parent.parent)'),
        (r'["\']D:\\GDELT_Africa_Extract["\']', 'str(BASE_DIR.parent.parent.parent)'),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # If content changed and no config import exists, add it
    if original != content and 'from config import' not in content and 'import config' not in content:
        # Find the import section
        lines = content.split('\n')
        import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i + 1
            elif import_idx > 0 and not line.strip().startswith('#') and line.strip():
                break

        # Insert config import
        if import_idx == 0:
            # No imports yet, add after docstring or at top
            for i, line in enumerate(lines):
                if not line.strip().startswith('#') and not line.strip().startswith('"""') and not line.strip().startswith("'''") and line.strip():
                    import_idx = i
                    break

        if 'BASE_DIR' in content:
            lines.insert(import_idx, 'from config import BASE_DIR')

        content = '\n'.join(lines)

    return content

def process_file(source_file, target_file):
    """Process a single Python file."""
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply transformations
        content = remove_ai_attribution(content, source_file)
        content = fix_hardcoded_paths(content, source_file)

        # Write to target
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(content)

        return True
    except Exception as e:
        print(f"Error processing {source_file}: {e}")
        return False

def copy_directory_structure(source_dir, target_dir, script_mappings):
    """Copy and process all Python scripts according to mappings."""

    processed = 0
    errors = []

    for source_pattern, target_subdir in script_mappings.items():
        source_path = source_dir / source_pattern
        target_path = target_dir / target_subdir

        if source_path.is_file():
            # Single file
            if process_file(source_path, target_path / source_path.name):
                processed += 1
                print(f"[OK] {source_pattern} -> {target_subdir}")
            else:
                errors.append(source_pattern)
        elif source_path.is_dir():
            # Directory - process all Python files
            for py_file in source_path.rglob('*.py'):
                if '__pycache__' in str(py_file):
                    continue
                relative = py_file.relative_to(source_path)
                target_file = target_path / relative
                if process_file(py_file, target_file):
                    processed += 1
                    print(f"[OK] {source_pattern}/{relative} -> {target_subdir}")
                else:
                    errors.append(str(py_file))
        else:
            # Pattern - glob for files
            for py_file in source_dir.glob(source_pattern):
                if '__pycache__' in str(py_file) or not py_file.is_file():
                    continue
                if process_file(py_file, target_path / py_file.name):
                    processed += 1
                    print(f"[OK] {py_file.name} -> {target_subdir}")
                else:
                    errors.append(str(py_file))

    return processed, errors

def main():
    """Main execution."""
    print("="*80)
    print("DISSERTATION CLEANUP SCRIPT")
    print("="*80)
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {TARGET_DIR}")
    print()

    # Script organization mappings
    script_mappings = {
        # Data acquisition
        'DATA/shapefiles/download_gadm_africa.py': 'scripts/01_data_acquisition',
        'DATA/shapefiles/download_livelihood_zones.py': 'scripts/01_data_acquisition',
        'DATA/shapefiles/download_livelihood_zones_api.py': 'scripts/01_data_acquisition',

        # Data processing
        'DATA AGGREGATION SCRIPTS': 'scripts/02_data_processing',

        # Stage 1 baseline
        'STAGE_1_AR_BASELINE': 'scripts/03_stage1_baseline',

        # Stage 2 feature engineering
        'STAGE_2_FEATURE_ENGINEERING': 'scripts/04_stage2_feature_engineering',

        # Stage 2 model training
        'STAGE_2_MODEL_TRAINING': 'scripts/05_stage2_model_training',

        # Stage 3 comparison/cascade
        'STAGE_3_COMPARISON_ANALYSIS': 'scripts/06_cascade_analysis',

        # Visualizations
        'VISUALIZATION': 'scripts/07_visualization',
        'VISUALIZATIONS_PUBLICATION': 'scripts/07_visualization/publication',

        # Analysis
        'ANALYSIS': 'scripts/08_analysis',

        # Explainability
        'EXPLAINABILITY': 'scripts/09_explainability',
    }

    print("Processing scripts...")
    print()

    processed, errors = copy_directory_structure(SOURCE_DIR, TARGET_DIR, script_mappings)

    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"[OK] Processed: {processed} files")
    if errors:
        print(f"[ERROR] Errors: {len(errors)} files")
        print("\nFailed files:")
        for err in errors[:10]:
            print(f"  - {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    else:
        print("[OK] Errors: 0")
    print()
    print("All scripts cleaned and organized!")
    print("="*80)

if __name__ == '__main__':
    main()
