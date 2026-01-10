"""
Fix imports in all processed scripts to use correct config.py paths.

Author: Victor Collins Oppon
MSc Data Science Dissertation, Middlesex University 2025
"""

import re
from pathlib import Path

def fix_config_imports(file_path):
    """Fix config imports to match new config.py structure."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Remove old incorrect import blocks
    patterns_to_remove = [
        r'from config import \(\s*BASE_DIR,\s*STAGE1_DATA_DIR.*?\)',
        r'from config import BASE_DIR, STAGE1_DATA_DIR.*?\n',
    ]

    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)

    # Check if config import exists at all
    if 'from config import' not in content and 'import config' not in content:
        # Add basic config import after other imports
        lines = content.split('\n')
        import_idx = -1

        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                import_idx = i + 1

        if import_idx > 0:
            # Find end of import block
            for i in range(import_idx, len(lines)):
                if lines[i].strip() and not lines[i].startswith('import ') and not lines[i].startswith('from ') and not lines[i].strip().startswith('#'):
                    import_idx = i
                    break

            lines.insert(import_idx, 'from config import BASE_DIR, INTERIM_DATA_DIR, RESULTS_DIR, N_FOLDS, RANDOM_SEED')
            lines.insert(import_idx + 1, '')
            content = '\n'.join(lines)

    # Replace hardcoded paths
    replacements = [
        # Stage 1 data paths
        (r"BASE_DIR / 'data' / 'district_level'", "INTERIM_DATA_DIR / 'stage1'"),
        (r"STAGE1_DATA_DIR", "INTERIM_DATA_DIR / 'stage1'"),
        (r"STAGE1_RESULTS_DIR", "RESULTS_DIR / 'stage1_baseline'"),

        # Stage 2 paths
        (r"STAGE2_FEATURES_DIR", "INTERIM_DATA_DIR / 'stage2'"),
        (r"STAGE2_MODELS_DIR", "RESULTS_DIR / 'stage2_models'"),
        (r"STAGE2_DATA_DIR", "INTERIM_DATA_DIR / 'stage2'"),

        # Results paths
        (r"RESULTS_DIR / 'STAGE_1'", "RESULTS_DIR / 'stage1_baseline'"),
        (r"RESULTS_DIR / 'STAGE_2'", "RESULTS_DIR / 'stage2_models'"),
        (r"RESULTS_DIR / 'CASCADE'", "RESULTS_DIR / 'cascade_optimized'"),

        # External data paths
        (r"BASE_DIR / 'DATA'", "BASE_DIR / 'data' / 'external'"),

        # Figure paths
        (r"FIGURES_DIR", "BASE_DIR / 'figures'"),
    ]

    for pattern, replacement in replacements:
        content = re.sub(pattern, replacement, content)

    if content != original:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    base_dir = Path('C:/GDELT_Africa_Extract/Scripts/district_pipeline/dissertation_submission')
    scripts_dir = base_dir / 'scripts'

    modified = 0
    for py_file in scripts_dir.rglob('*.py'):
        if fix_config_imports(py_file):
            modified += 1
            print(f"[FIXED] {py_file.relative_to(base_dir)}")

    print(f"\n[OK] Fixed {modified} files")

if __name__ == '__main__':
    main()
