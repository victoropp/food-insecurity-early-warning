from pathlib import Path
from datetime import datetime, timedelta

figures_dir = Path('C:/GDELT_Africa_Extract/Scripts/district_pipeline/FINAL_PIPELINE/FIGURES')
cutoff = datetime.now() - timedelta(hours=1)

updated = [(f.parent.name, datetime.fromtimestamp(f.stat().st_mtime)) for f in figures_dir.rglob('*.png')]
categories = {}
for cat, date in updated:
    if cat not in categories:
        categories[cat] = []
    categories[cat].append(date)

recent = {cat: max(dates) for cat, dates in categories.items()}

print("\nFigure Categories - Update Status (last hour):\n")
for cat in sorted(recent.keys()):
    is_recent = recent[cat] > cutoff
    status = 'RECENT' if is_recent else 'OLD'
    print(f'{status:8} {cat:30} {recent[cat].strftime("%Y-%m-%d %H:%M")}')

print(f"\n{len([c for c in recent.values() if c > cutoff])} categories updated in last hour")
print(f"{len(categories)} total categories")
