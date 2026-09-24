#!/usr/bin/env python3
"""Submit after the pilot acceptance log exists; keep scoring within 64 auxiliary cores."""
import json
import subprocess
from pathlib import Path

ROOT=Path('/data/islamm/retention_leakage')
log=(ROOT/'build/logs/fx9_5_pilot_check.log').read_text()
assert log.count('PILOT ACCEPT')==4 and 'Traceback' not in log,log
for method in ['m1_glfc','m2_target','m4_proto','m5_hybrid_replay']:
    assert f'PILOT ACCEPT {method} imagenet_r loadable=64/64' in log
out=ROOT/'build/waves/fx9_job_ids.json'
assert not out.exists(),'Wave jobs already submitted; inspect recorded IDs instead of duplicating them'
jobs={}


def submit(name,args):
    result=subprocess.run(['qsub',*args],cwd=ROOT,text=True,capture_output=True,check=True)
    job=result.stdout.strip()
    assert '.bcm11' in job,(result.stdout,result.stderr)
    jobs[name]=job
    out.write_text(json.dumps(jobs,indent=2)+'\n')
    print(name,job,flush=True)
    return job


def generic(name,script,cores,mem,depends,array=None):
    args=['-N',name,'-l',f'select=1:ncpus={cores}:mem={mem}gb','-l','walltime=24:00:00',
          '-W','depend=afterok:'+':'.join(depends),'-o',str(ROOT/'build/logs'/(name+'.log')) if array is None else str(ROOT/'build/logs/'),
          '-v','JOBFILE='+str(ROOT/'build/jobs'/script)]
    if array is not None:
        args+=['-J',array]
    return submit(name,args+['code/scripts/pbs/run_job.pbs'])


v3=submit('v3',['-N','fx9_v3','-J','0-575%40','code/scripts/pbs/shadow_v3.pbs'])
dose=submit('dose',['-N','fx9_dose','-J','0-287%16','code/scripts/pbs/fx9_dose_response.pbs'])
v3check=generic('fx9_v3_verify','fx9_wave_verify.sh',1,8,[v3])
dosecheck=generic('fx9_dose_verify','fx9_dose_verify.sh',1,8,[dose])
# All currently staged auxiliary runs finish before these 36+24-core scoring arrays.
# The two shadow waves need not finish together; their own load check is mandatory.
aux=['184371[].bcm11','184365.bcm11','184366.bcm11','184372[].bcm11','184373.bcm11','184360.bcm11']
score=generic('fx9_score','fx9_score.sh',2,8,[v3check,*aux],array='0-17%18')
dosescore=generic('fx9_dose_score','fx9_dose_score.sh',2,8,[dosecheck,*aux],array='0-35%12')
final=generic('fx9_final','fx9_final.sh',4,16,[score,dosescore])
with (ROOT/'build/STATE.md').open('a') as f:
    f.write('\n### FX9 production wave submissions\n\nPilot acceptance verified before qsub; 4 methods × 64 loadable files.\n\n```json\n'+json.dumps(jobs,indent=2)+'\n```\n\nV3: 320 cores; dose shadows:128; current auxiliaries≤56. Scoring waits for listed auxiliary jobs and its wave load check; score arrays then use36+24=60 cores. Final4-core job depends on both scoring arrays.\n')
