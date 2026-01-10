"""
Clean all old visualizations from dissertation figures folder
Keep only middlesex_logo.png
"""
import pathlib
from config import BASE_DIR

figures_dir = pathlib.Path(rstr(BASE_DIR))

count = 0
for f in figures_dir.rglob('*'):
    if f.is_file() and f.suffix in ['.pdf', '.png'] and f.name != 'middlesex_logo.png':
        print(f"Deleting: {f.name}")
        f.unlink()
        count += 1

print(f"\n[OK] Deleted {count} old visualization files")
print("[OK] Clean slate ready for new publication-grade figures")
