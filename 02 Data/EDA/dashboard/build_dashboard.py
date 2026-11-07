"""Embed the analysis data in an offline dashboard."""
import json
from pathlib import Path

folder = Path(__file__).resolve().parent
analysis = json.loads((folder / 'analysis.json').read_text(encoding='utf-8'))
payload = json.dumps(analysis, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
template = (folder / 'template.html').read_text(encoding='utf-8')
if template.count('__DATA_JSON__') != 1:
    raise ValueError('Expected exactly one data placeholder')
(folder / 'dashboard.html').write_text(template.replace('__DATA_JSON__', payload), encoding='utf-8')
print(f"Built dashboard.html with {analysis['kpis']['rows']:,} fitting-period rows")
