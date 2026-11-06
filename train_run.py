import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import train
from configs import CONFIGS

if __name__ == '__main__':
    name, seed = sys.argv[1], int(sys.argv[2])
    extra = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    os.makedirs('results/models', exist_ok=True)
    cfg = dict(CONFIGS[name], **extra, name=name, seed=seed)
    out = f'results/models/{name}_s{seed}'
    if os.path.exists(out + '_actor.npz'):
        print('exists', out); sys.exit(0)
    train(cfg, out_prefix=out)
