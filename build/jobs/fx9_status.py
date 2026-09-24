import json
import subprocess
from datetime import datetime
from pathlib import Path

root=Path('/data/islamm/retention_leakage')
jobs=json.loads((root/'build/waves/fx9_job_ids.json').read_text())
ids=[jobs[k] for k in ['v3','dose','fx9_v3_verify','fx9_dose_verify','fx9_score','fx9_dose_score','fx9_final']]+[jobs.get('fx9_preserved','184371[].bcm11'),jobs.get('fx9_wave_rebuild','184455.bcm11')]
r=subprocess.run(['qstat','-xf','-F','json',*ids],text=True,capture_output=True)
if r.returncode:
 print(r.stderr)
d=json.loads(r.stdout)
print(datetime.now().astimezone().strftime('%F %T %Z'))
for k,v in d['Jobs'].items():
 print(k, v['Job_Name'], v['job_state'], v.get('array_state_count',''), 'exit='+str(v.get('Exit_status','pending')))
errors=[k for k,v in d['Jobs'].items() if v.get('Exit_status',0)!=0]
if errors:
 print('ERROR JOBS', errors)
