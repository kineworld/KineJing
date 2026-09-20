"""Explicit action/feature contract for neural dynamics; no implicit normalization."""
import json
from pathlib import Path
import numpy as np

def validate_schema(schema, width):
    if not isinstance(schema,dict): raise ValueError('Action schema must be an object')
    columns=schema.get('columns');units=schema.get('units')
    if not isinstance(columns,list) or len(columns)!=width or any(not isinstance(x,str) or not x.strip() for x in columns):
        raise ValueError('Action columns must name every dimension')
    if len(set(columns))!=width: raise ValueError('Action column names must be unique')
    if not isinstance(units,list) or len(units)!=width or any(not isinstance(x,str) or not x.strip() for x in units):
        raise ValueError('Declare a unit for each action dimension')
    for key in ['coordinate_frame','normalization']:
        if not isinstance(schema.get(key),str) or not schema[key].strip(): raise ValueError('Missing '+key)

def validate_inputs(tokens,actions,metadata,schema,checkpoint=None):
    if tokens.ndim!=3 or actions.ndim!=3 or any(n<1 for n in tokens.shape+actions.shape) or tokens.shape[0]!=actions.shape[0]:
        raise ValueError('Expected nonempty tokens B,N,D and actions B,H,A with matching batch')
    if not np.isfinite(tokens).all() or not np.isfinite(actions).all(): raise ValueError('Non-finite input')
    if not isinstance(metadata,dict) or any(not isinstance(metadata.get(k),str) or not metadata[k] for k in ['encoder','checkpoint_sha256']):
        raise ValueError('Feature provenance required')
    validate_schema(schema,actions.shape[-1])
    if checkpoint is not None:
        config=checkpoint['config']
        if tokens.shape[-1]!=config['dim'] or actions.shape[-1]!=config.get('action_dim',8): raise ValueError('Checkpoint dimension mismatch')
        if metadata!=checkpoint.get('encoder_provenance'): raise ValueError('Encoder provenance mismatch')
        if schema!=checkpoint.get('action_schema'): raise ValueError('Action convention differs from training checkpoint')

def pack(features,actions_file,schema_file,output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    with np.load(features,allow_pickle=False) as source:
        tokens=source['tokens'];metadata=json.loads(str(source['metadata'].item()))
    actions=np.load(actions_file,allow_pickle=False)
    if not isinstance(actions,np.ndarray): raise ValueError('Actions must be a .npy array')
    schema=json.loads(Path(schema_file).read_text(encoding='utf-8'))
    validate_inputs(tokens,actions,metadata,schema)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as f:
        np.savez_compressed(f,tokens=tokens,actions=actions,metadata=np.array(json.dumps(metadata)),action_schema=np.array(json.dumps(schema)))
    return {'output':str(output),'tokens_shape':list(tokens.shape),'actions_shape':list(actions.shape),'normalization_applied':False}
