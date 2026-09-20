import argparse,json,sys
from pathlib import Path
from .adapters import registry,plan,execute


def main():
    if len(sys.argv)>1 and sys.argv[1]=='triworld':
        from .triworld import main as generate
        sys.argv=[sys.argv[0]]+sys.argv[2:]
        return generate()
    p=argparse.ArgumentParser(description='KineJing / 勘境: composable world-model research')
    subs=p.add_subparsers(dest='command',required=True)
    subs.add_parser('catalog',help='Show backends, pinned versions and verification status')
    d=subs.add_parser('demo',help='Run a synthetic CPU demo without model downloads');d.add_argument('--output',type=Path,required=True)
    l=subs.add_parser('launch',help='Run one explicitly configured external backend');l.add_argument('--config',type=Path,required=True);l.add_argument('--dry-run',action='store_true')
    b=subs.add_parser('pack-actions',help='Connect encoder tokens with explicitly documented robot actions')
    for key in ['features','actions','schema','output']:b.add_argument('--'+key,type=Path,required=True)
    r=subs.add_parser('rank',help='Compare compatible features to a user-supplied goal');r.add_argument('--goal',type=Path,required=True);r.add_argument('--candidates',type=Path,nargs='+',required=True);r.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.command=='catalog':result=registry()
    elif a.command=='demo':
        from .demo import demo
        result=demo(a.output)
    elif a.command=='launch':
        config=json.loads(a.config.read_text(encoding='utf-8'));result=plan(config) if a.dry_run else execute(config)
    elif a.command=='pack-actions':
        from .rollout_io import pack
        result=pack(a.features,a.actions,a.schema,a.output)
    else:
        from .features import rank
        result=rank(a.goal,a.candidates)
        if a.output:
            if a.output.exists():raise FileExistsError(a.output)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2,ensure_ascii=False))
