"""Compare 8 versus 64 paired training maps per teammate under the original LBF CRP objective."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import json
from multiprocessing import get_context
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_lbf_crp import AuditCRP, FORMAL, file_hash
from scripts.lbf_crp_heldout import collect_heldout, evaluate_heldout
from fastap_methods.fastap_crp import BehaviorModel


def fit_seed(seed, training, members, model_args, config):
    torch.set_num_threads(1)
    data = {key: torch.from_numpy(batch) for key, batch in training.items()}
    runs = []
    for maps in (8, 64):
        torch.manual_seed(seed)
        crp = AuditCRP('paper', **model_args, alpha=config['alpha'], lr=config['lr'],
                       minibatch_size=config['minibatch_size'], device='cpu')
        for i, name in enumerate(members):
            rows = slice(i, maps * len(members), len(members))
            episode_batch = {k: data[k][rows] for k in ('state', 'teammate_actions', 'teammate_mask', 'filled')}
            crp.add_group(name, episode_batch, model_steps=config['crp_model_steps'])
        runs.append((dict(seed=seed, maps_per_member=maps,
                          assignments={name: f'C{label+1}' for name, label in crp.assignments.items()},
                          updates=len(crp.max_weight), updates_above_99pct=sum(w > .99 for w in crp.max_weight),
                          median_effective_batch_fraction=float(np.median(crp.effective_batch_fraction))),
                     {name: t.numpy() for name, t in crp.model.state_dict().items()}))
    return runs


def run(output):
    source = json.loads((FORMAL / 'clustering.json').read_bytes())
    pool = ROOT / source['pool_manifest']
    if file_hash(pool) != source['pool_manifest_sha256']:
        raise ValueError("The source pool manifest changed since the formal run")
    saved = torch.load(FORMAL / 'crp.pt', map_location='cpu', weights_only=True)
    formal_probes = saved['crp']['probes']
    train = collect_heldout(pool, formal_probes, map_seeds=range(982000, 982064), ego_seed_start=419000)
    members = train['members']
    batch = train['trials']['fixed_T1_ego']
    training = {k: t.numpy() for k, t in batch.items()}
    seeds = (7, 17, 27)
    # NumPy payloads keep workers independent of PyTorch's interprocess tensor storage.
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn')) as workers:
        jobs = [workers.submit(fit_seed, seed, training, members, saved['model_args'], source['protocol'])
                for seed in seeds]
        runs = [run for job in jobs for run in job.result()]
    # Every member shares each start; one full probe per map suffices to exclude it.
    probes_per_map = zip(*(batch[k][::len(members)] for k in
                           ('state', 'teammate_actions', 'teammate_mask', 'filled')))
    held = collect_heldout(pool, chain(formal_probes, probes_per_map), map_seeds=range(981000, 981032),
                           ego_seed_start=418000)
    path = pool.with_name('crossplay.json')
    crossplay = json.loads(path.read_bytes())
    results = []
    for stats, weights in runs:
        model = BehaviorModel(**saved['model_args'])
        model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
        results.append(dict(**stats, **evaluate_heldout(model, stats['assignments'], held, crossplay)))
    returns = {}
    for maps in (8, 64):
        scores = batch['reward'][:maps * len(members)].sum(1).double().numpy()
        returns[str(maps)] = dict(episodes=len(scores), mean_return=float(scores.mean()),
                                  any_food=int((scores > 0).sum()), all_food=int(np.isclose(scores, 1.).sum()))
    report = dict(training_ego='Same frozen original T1 ego pair for all members; same initial maps across members.',
                  design='Eight-map arm is the prefix of the 64-map arm. Both use the same per-group update budget.',
                  interpretation='More trajectories enter both the model fit and the summed MAP log evidence.',
                  training_map_seeds=train['map_seeds'], heldout_map_seeds=held['map_seeds'],
                  heldout_random_ego_seed_start=held['ego_seed_start'], seeds=seeds, member_order=members,
                  model_steps_per_member=source['protocol']['crp_model_steps'], alpha=source['protocol']['alpha'],
                  lr=source['protocol']['lr'], minibatch_size=source['protocol']['minibatch_size'],
                  formal_checkpoint_sha256=file_hash(FORMAL/'crp.pt'), pool_manifest_sha256=file_hash(pool),
                  crossplay_sha256=file_hash(path), training_returns=returns, results=results)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?',
                        default=FORMAL.parent/'crp_diagnostics/figure6_audit/coverage.json')
    args = parser.parse_args()
    output = (ROOT/args.output).resolve()
    if os.getenv('FASTAP_ROOT') != str(ROOT) or ROOT/'outputs' not in output.parents:
        parser.error("Source scripts/epymarl-env.sh and choose an output file under project outputs/")
    if output.exists():
        parser.error("Choose a fresh report file")
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    torch.set_num_threads(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    run(output)
