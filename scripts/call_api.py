"""Call the local API using authorized TriWorldBench-format input files."""
import argparse
import base64
import json
import os
from pathlib import Path

import h5py
import httpx


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--episode',type=int,default=1)
    p.add_argument('--url',default='http://127.0.0.1:8765')
    a=p.parse_args()
    key=os.environ.get('KINEJING_API_KEY')
    if not key:raise ValueError('Set KINEJING_API_KEY locally; do not paste it into an issue')
    with h5py.File(a.inputs/'data'/f'episode{a.episode}.hdf5') as f:
        actions=f['joint_action/vector'][:].tolist()
    images={view:base64.b64encode((a.inputs/'first_frame'/f'episode{a.episode}_{view}.jpg').read_bytes()).decode()
            for view in ('head','left','right')}
    with httpx.Client(timeout=120,trust_env=False) as client:
        result=client.post(a.url.rstrip('/')+'/v1/predictions',
            headers={'Authorization':'Bearer '+key},json={
                'action_schema':'triworldbench-joint-action-vector-v1','images':images,'actions':actions})
        result.raise_for_status();data=result.json()
    print(json.dumps({k:data[k] for k in ('request_id','model','output_kind','shape','inference_seconds')},indent=2))

if __name__=='__main__':main()
