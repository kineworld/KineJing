"""Pinned external launchers. No shell, silent fallbacks or implicit downloads."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def registry():
    return json.loads(Path(__file__).with_name('backends.json').read_text(encoding='utf-8'))


def plan(config):
    kind=config['backend']
    if kind not in ('causalwm','wan22','vjepa21','kine-jepa'):
        raise ValueError('Unknown external backend: '+str(kind))
    if config.get('actions') and kind in ('causalwm','wan22','vjepa21'):
        raise ValueError('This released backend does not consume robot actions; do not silently drop them.')
    repo=Path(config['repo']).resolve()
    python=str(config.get('python',sys.executable))
    output=Path(config['output']).resolve()
    required=[]
    def path(key):
        p=Path(config[key]).resolve();required.append(p);return str(p)
    if kind=='causalwm':
        script=repo/'inference.py';required.append(script)
        frames=int(config.get('frames',121));width=int(config.get('width',640));height=int(config.get('height',480))
        if frames<9 or (frames-1)%8 or width<=0 or height<=0 or width%32 or height%32:
            raise ValueError('CausalWM needs 8k+1 frames and positive dimensions divisible by 32.')
        argv=[python,str(script),'--image',path('image'),'--prompt',config['prompt'],
              '--checkpoint',path('checkpoint'),'--base-ckpt',path('base_checkpoint'),
              '--text-encoder-dir',path('text_encoder'),'--out-dir',str(output),
              '--frames',str(frames),'--width',str(width),'--height',str(height),'--seed',str(config.get('seed',42))]
    elif kind=='wan22':
        script=repo/'generate.py';required.append(script)
        frames=int(config.get('frames',81))
        if frames<5 or (frames-1)%4: raise ValueError('Wan needs 4k+1 frames.')
        argv=[python,str(script),'--task','ti2v-5B','--size','1280*704',
              '--ckpt_dir',path('checkpoint'),'--image',path('image'),
              '--prompt',config['prompt'],'--frame_num',str(frames),'--save_file',str(output),
              '--offload_model','True','--t5_cpu','--convert_model_dtype',
              '--base_seed',str(config.get('seed',42))]
    else:
        script=Path(__file__).with_name('neural.py').resolve()
        argv=[python,str(script),kind,'--repo',str(repo),'--checkpoint',path('checkpoint'),
              '--input',path('input'),'--output',str(output),'--device',config.get('device','cpu')]
    return {'backend':kind,'cwd':str(repo),'argv':argv,'output':str(output),
            'required':[str(x) for x in required], 'expected_revision':registry()[kind]['revision']}


def execute(config):
    job=plan(config)
    for p in job['required']:
        if not Path(p).exists(): raise FileNotFoundError(p)
    actual=subprocess.check_output(['git','-C',job['cwd'],'rev-parse','HEAD'],text=True).strip()
    if actual!=job['expected_revision']:
        raise ValueError('Upstream revision differs from backends.json. Check out the pinned commit.')
    dest=Path(job['output'])
    if dest.exists(): raise FileExistsError('Use a new output path: '+str(dest))
    dest.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(job['argv'],cwd=job['cwd'],check=True)
    if not dest.exists(): raise RuntimeError('Backend exited without producing the requested output.')
    receipt={'backend':job['backend'],'upstream_revision':actual,'executed':True,
             'official_benchmark_score':None,'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()}
    Path(str(dest)+'.run.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt
