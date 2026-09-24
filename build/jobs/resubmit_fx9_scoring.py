import json
import subprocess
from pathlib import Path

ROOT=Path('/data/islamm/retention_leakage')
path=ROOT/'build/waves/fx9_job_ids.json'
jobs=json.loads(path.read_text())
assert 'superseded_scoring' not in jobs
jobs['superseded_scoring']={k:jobs[k] for k in ['fx9_score','fx9_dose_score','fx9_final']}


def submit(name,script,cores,mem,depends,array=None):
    args=['qsub','-N',name,'-l',f'select=1:ncpus={cores}:mem={mem}gb','-l','walltime=24:00:00',
          '-W','depend=afterok:'+':'.join(depends),'-o',str(ROOT/'build/logs'/(name+'.log')) if array is None else str(ROOT/'build/logs/'),
          '-v','JOBFILE='+str(ROOT/'build/jobs'/script)]
    if array:
        args+=['-J',array]
    args+=['code/scripts/pbs/run_job.pbs']
    r=subprocess.run(args,cwd=ROOT,text=True,capture_output=True,check=True)
    job=r.stdout.strip()
    assert '.bcm11' in job
    jobs[name]=job
    path.write_text(json.dumps(jobs,indent=2)+'\n')
    print(name,job,flush=True)
    return job


render=submit('fx9_m9_render','fx9_m9_render.sh',1,4,['184373.bcm11'])
aux=['184371[].bcm11','184365.bcm11','184418.bcm11','184372[].bcm11','184373.bcm11','184360.bcm11']
score=submit('fx9_score','fx9_score.sh',2,8,[jobs['fx9_v3_verify'],*aux],'0-17%18')
dose=submit('fx9_dose_score','fx9_dose_score.sh',2,8,[jobs['fx9_dose_verify'],*aux],'0-35%12')
submit('fx9_final','fx9_final.sh',4,16,[score,dose,render])
with (ROOT/'build/STATE.md').open('a') as f:
    f.write('\nFIG13 render metadata serialization failed (numpy float figure dimensions). Fixed by casting to Python float; repair184418 exited0, 39 cells×25trials, no experiment rerun. PBS automatically cancelled dependent score/final jobs184414/184415/184416; replacements and a separate M9 render are:\n\n```json\n'+json.dumps(jobs,indent=2)+'\n```\n')
