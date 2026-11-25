"""CRP analysis of the published T1–T8 LBF actors, using the training probe collector."""

from contextlib import closing
from copy import deepcopy
import hashlib
import json
from math import log
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import yaml

from fastap_scenarios.fastap_rollout import FastapRolloutEnv, load_fastap_groups
from .fastap_crp import TeammateCRP, _collate
from .fastap_training import ROOT, _batch, collect_probe_episodes


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _candidate_scores(counts, alpha, scores):
    """Include the common CRP denominator in the reported log prior and joint."""
    log_normalizer = log(sum(counts) + alpha)
    rows = []
    for index, (weight, score) in enumerate(zip([*counts, alpha], scores)):
        log_weight = log(weight)
        rows.append(dict(cluster=f"C{index + 1}", is_new=index == len(counts),
                         prior_count=counts[index] if index < len(counts) else 0,
                         log_prior=log_weight - log_normalizer,
                         log_evidence=score - log_weight,
                         log_joint=score - log_normalizer))
    return rows


@torch.no_grad()
def _final_embeddings(crp, probes, admissions):
    """Every point uses the encoder fitted through the last admission."""
    crp.model.eval()
    rows = []
    for event in admissions:
        start, end = event['probe_slice']
        group = probes[start:end]
        vectors = crp.model.encode(*_collate(group, crp.device)).tolist()
        for i, (probe, vector) in enumerate(zip(group, vectors), start=1):
            rows.append(dict(teammate=event['teammate'], cluster=event['cluster'],
                             episode=i, length=len(probe[1]), embedding=vector))
    return dict(encoder="final", dimension=crp.model.width, rows=rows)


def _food_outcomes(returns):
    scores = np.asarray(returns, dtype=np.float64)
    return dict(episodes=len(returns), any_food=int((scores > 0).sum()),
                all_food=int(np.isclose(scores, 1.).sum()), mean_return=float(scores.mean()))


def run_crp(pool_path, output_dir, *, seed=7, method_config=None, probe_ego='own_original',
            probe_seed=None, decoder=None, masked_action_weight=None, masked_action_span=None):
    """Run standalone, sequential MAP clustering on CPU and publish its audit artifacts."""
    pool_path = (ROOT / pool_path).resolve()
    output_dir = (ROOT / output_dir).resolve()
    if ROOT not in output_dir.parents or ROOT not in pool_path.parents:
        raise ValueError("Keep the pool and CRP artifacts inside the project")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if probe_ego not in ('own_original', 'random_legal'):
        raise ValueError("probe_ego must be own_original or random_legal")
    if method_config is None:
        method_config = yaml.safe_load((ROOT / 'configs/methods/fastap.yaml').read_text())
    cfg = {**method_config, **method_config['scenes']['lbf']}
    decoder = cfg.get('crp_decoder', 'paper') if decoder is None else decoder
    if masked_action_weight is None:
        masked_action_weight = cfg.get('masked_action_weight', 0.)
    if masked_action_span is None:
        masked_action_span = cfg.get('masked_action_span', 3)
    for key in ('probe_episodes', 'crp_model_steps', 'groups_per_round'):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")

    manifest_bytes = pool_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    # load_fastap_groups checks each actor's SHA-256 and its scene, shape and role.
    groups = [g for g in load_fastap_groups('lbf', pool_path) if len(g.policies) == 1]
    if pool_path.read_bytes() != manifest_bytes:
        raise ValueError("The teammate pool changed while loading its policies")
    order = [g.name for g in groups]
    if manifest.get('target_size') != 8 or order != [f'T{i}' for i in range(1, 9)]:
        raise ValueError("Expected the complete LBF pool in T1–T8 publication order")
    use_original_egos = probe_ego == 'own_original'
    if use_original_egos:
        for group in groups:
            if not group.probe_egos or len(group.probe_egos) != 2:
                raise ValueError(f"Missing original QMIX partners for {group.name}; expected roles 0 and 1")

    probe_seed = seed + 1 if probe_seed is None else probe_seed
    rng = np.random.default_rng(probe_seed)
    admissions, snapshots, assignments, clusters = [], [], {}, {}
    probe_outcomes = {}
    with closing(FastapRolloutEnv('lbf', groups, seed=probe_seed)) as env, torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model_args = dict(state_dim=env.state_shape, n_actions=env.n_actions, n_teammates=env.max_teammates)
        crp = TeammateCRP(**model_args, alpha=cfg['alpha'], lr=cfg['lr'], device='cpu', decoder=decoder,
                          masked_action_weight=masked_action_weight, masked_action_span=masked_action_span,
                          mask_seed=seed)
        for i, group in enumerate(groups):
            episodes = collect_probe_episodes(env, i, rng, cfg['probe_episodes'],
                                              use_original_egos=use_original_egos)
            returns = [ep['reward'][ep['filled']].sum().item() for ep in episodes]
            probe_outcomes[group.name] = dict(episode_returns=returns, **_food_outcomes(returns))
            counts = list(crp.counts)
            centers = list(crp.centers)  # add_group replaces a center after each MAP decision.
            label = crp.add_group(group.name, _batch(episodes), model_steps=cfg['crp_model_steps'])
            cluster = f"C{label + 1}"
            assignments[group.name] = cluster
            clusters.setdefault(cluster, []).append(group.name)
            start = i * cfg['probe_episodes']
            probe_slice = [start, start + len(episodes)]
            admissions.append(dict(teammate=group.name, round=1 + i // cfg['groups_per_round'],
                                   cluster=cluster, probe_slice=probe_slice,
                                   last_model_loss=crp.last_model_loss,
                                   candidates=_candidate_scores(counts, crp.alpha,
                                                                crp.last_assignment_log_scores.tolist())))
            if masked_action_weight:
                admissions[-1].update(last_reconstruction_loss=crp.last_reconstruction_loss,
                                      last_masked_action_loss=crp.last_masked_action_loss)
            # Each snapshot can replay that exact assignment on the stored probe slice,
            # even after subsequent fitting changes the encoder and decoder parameters.
            snapshots.append(dict(teammate=group.name, probe_slice=probe_slice, counts_before=counts,
                                  centers_before=centers, model=deepcopy(crp.model.state_dict())))
        final = crp.state_dict()
        embeddings = _final_embeddings(crp, final['probes'], admissions)
        protocol = dict(seed=seed, probe_env_seed=probe_seed, probe_ego_seed=probe_seed,
                        order=order, groups_per_round=cfg['groups_per_round'], alpha=crp.alpha,
                        probe_episodes=cfg['probe_episodes'], crp_model_steps=cfg['crp_model_steps'],
                        lr=cfg['lr'], minibatch_size=crp.minibatch_size, model_args=model_args,
                        episode_limit=env.episode_limit, probe_ego=probe_ego, decoder=decoder,
                        training_objective=('mean of within-episode mean categorical action NLL'
                                            if decoder == 'categorical' else 'Eq. 4 Gaussian trajectory mixture'),
                        admission_evidence='sum of action log likelihoods over observed slots, steps and episodes',
                        ego_policy="original greedy QMIX partners" if use_original_egos else "uniform legal actions",
                        ego_policy_source=("roles 0 and 1 from the same verified frozen team checkpoint as role 2"
                                           if use_original_egos else "sampled uniformly from each ego's legal actions"),
                        teammate_role=manifest['role'], active_teammates=1,
                        device='cpu', torch_threads=torch.get_num_threads(),
                        torch_version=str(torch.__version__), numpy_version=np.__version__)
        if masked_action_weight:
            protocol.update(masked_action_weight=masked_action_weight, masked_action_span=masked_action_span,
                            mask_seed=seed, training_objective='full episode categorical NLL plus weighted '
                            'categorical NLL on a hidden action span of the same episode; both episode-balanced')
    all_returns = [value for item in probe_outcomes.values() for value in item['episode_returns']]
    summary = dict(pool_manifest=str(pool_path.relative_to(ROOT)),
                   pool_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                   member_sha256={m['id']: m['sha256'] for m in manifest['members']},
                   protocol=protocol, assignments=assignments, clusters=clusters,
                   admissions=admissions, probe_outcomes=dict(per_group=probe_outcomes, total=_food_outcomes(all_returns)))
    output_dir.parent.mkdir(exist_ok=True, parents=True)
    with TemporaryDirectory(dir=output_dir.parent, prefix=f".{output_dir.name}-") as staging_dir:
        staging = Path(staging_dir)
        _write_json(staging / 'clustering.json', summary)
        _write_json(staging / 'embeddings.json', embeddings)
        torch.save(dict(model_args=model_args, crp=final, admissions=snapshots), staging / 'crp.pt')
        staging.rename(output_dir)
    return summary
