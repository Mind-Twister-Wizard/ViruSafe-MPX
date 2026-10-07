from __future__ import annotations
from pathlib import Path
import compileall, json, sys
required=[
    'run_pipeline.py','run_v11.py','config.yaml','config_v1.1.yaml',
    'src/pipeline.py','src_v11/pipeline.py','README.md',
    'results/summaries/results_summary_v1.1.json'
]
missing=[p for p in required if not Path(p).exists()]
if missing:
    print('Missing:', *missing, sep='\n - '); raise SystemExit(2)
ok=compileall.compile_dir('src',quiet=1) and compileall.compile_dir('src_v11',quiet=1)
json.loads(Path('results/summaries/results_summary_v1.1.json').read_text(encoding='utf-8'))
print('Repository validation:', 'PASS' if ok else 'FAIL')
raise SystemExit(0 if ok else 1)
