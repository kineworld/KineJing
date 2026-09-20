"""Compare compatible goal/candidate features; this is not TWB-Score."""
import json
from pathlib import Path
import numpy as np


def load_features(path):
    with np.load(path,allow_pickle=False) as bundle:
        embedding=np.asarray(bundle['embedding'],dtype=np.float64).reshape(-1)
        metadata=json.loads(str(bundle['metadata'].item()))
    if not embedding.size or not np.all(np.isfinite(embedding)) or np.linalg.norm(embedding)<1e-12:
        raise ValueError('Feature vector must be finite and nonzero.')
    for key in ['encoder','checkpoint_sha256']:
        if not isinstance(metadata.get(key),str) or not metadata[key]: raise ValueError('Missing provenance: '+key)
    return embedding,metadata


def rank(goal,candidates):
    g,meta=load_features(goal);rows=[]
    for candidate in candidates:
        x,other=load_features(candidate)
        if x.shape!=g.shape or any(meta[k]!=other[k] for k in ['encoder','checkpoint_sha256']):
            raise ValueError('Cannot compare incompatible feature spaces: '+str(candidate))
        score=float(np.clip(np.dot(x,g)/(np.linalg.norm(x)*np.linalg.norm(g)),-1,1))
        rows.append({'candidate':str(candidate),'goal_cosine':score})
    return {'metric':'goal feature cosine; not official TWB-Score','results':sorted(rows,key=lambda r:-r['goal_cosine'])}
