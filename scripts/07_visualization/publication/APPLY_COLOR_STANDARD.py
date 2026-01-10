"""
Apply COLOR_SCHEME_STANDARD.md to all visualization files
Updates existing visualization scripts to use consistent country colors

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import re
from pathlib import Path
from config import BASE_DIR

# STANDARD COLOR SCHEME
STANDARD_COLORS = {
    'Zimbabwe': '#E74C3C',  # Red
    'Sudan': '#3498DB',     # Blue
    'DRC': '#9B59B6',       # Purple
}

# Files to update
VISUALIZATION_DIR = Path(str(BASE_DIR))

FILES_TO_UPDATE = [
    'ch04_key_saves_map.py',
    'ch04_cascade_real_stories.py',
    'ch05_cascade_breakthrough.py',
    'ch05_cascade_geographic_map.py',
]

# Color patterns to find and replace
COLOR_PATTERNS = {
    'Zimbabwe': [
        (r"'#[0-9A-F]{6}'.*Zimbabwe", f"'{STANDARD_COLORS['Zimbabwe']}'  # Zimbabwe (RED - STANDARD)"),
        (r'"#[0-9A-F]{6}".*Zimbabwe', f'"{STANDARD_COLORS["Zimbabwe"]}"  # Zimbabwe (RED - STANDARD)'),
    ],
    'Sudan': [
        (r"'#[0-9A-F]{6}'.*Sudan", f"'{STANDARD_COLORS['Sudan']}'  # Sudan (BLUE - STANDARD)"),
        (r'"#[0-9A-F]{6}".*Sudan', f'"{STANDARD_COLORS["Sudan"]}"  # Sudan (BLUE - STANDARD)'),
    ],
    'DRC': [
        (r"'#[0-9A-F]{6}'.*DRC", f"'{STANDARD_COLORS['DRC']}'  # DRC (PURPLE - STANDARD)"),
        (r'"#[0-9A-F]{6}".*DRC', f'"{STANDARD_COLORS["DRC"]}"  # DRC (PURPLE - STANDARD)'),
    ],
}

def apply_standard_colors(file_path):
    """Apply standard colors to a visualization file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        original_content = content
        changes_made = 0

        # Apply color replacements
        for country, patterns in COLOR_PATTERNS.items():
            for pattern, replacement in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    changes_made += len(matches)

        # Write back if changes were made
        if changes_made > 0:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True, changes_made
        else:
            return False, 0

    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return False, 0

def main():
    """Apply standard colors to all visualization files"""
    print("="*80)
    print("APPLYING COLOR_SCHEME_STANDARD.md TO VISUALIZATION FILES")
    print("="*80)

    total_updated = 0
    total_changes = 0

    for filename in FILES_TO_UPDATE:
        file_path = VISUALIZATION_DIR / filename

        if file_path.exists():
            updated, changes = apply_standard_colors(file_path)
            if updated:
                print(f"[OK] {filename}: {changes} color(s) updated")
                total_updated += 1
                total_changes += changes
            else:
                print(f"[SKIP] {filename}: Already using standard colors")
        else:
            print(f"[ERROR] {filename}: File not found")

    print("\n" + "="*80)
    print(f"SUMMARY: {total_updated} file(s) updated, {total_changes} total color change(s)")
    print("="*80)

    if total_updated > 0:
        print("\n[OK] Standard colors applied successfully!")
        print("\nStandard Color Scheme:")
        print(f"  - Zimbabwe: {STANDARD_COLORS['Zimbabwe']} (RED)")
        print(f"  - Sudan: {STANDARD_COLORS['Sudan']} (BLUE)")
        print(f"  - DRC: {STANDARD_COLORS['DRC']} (PURPLE)")

if __name__ == '__main__':
    main()
