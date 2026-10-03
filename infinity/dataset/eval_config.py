"""Scene and view selections shared by training evaluation loaders."""

import json
from pathlib import Path


def load_eval_config(json_path):
    with open(json_path) as source:
        config = json.load(source)
    if not isinstance(config, dict):
        raise ValueError(f'{json_path}: evaluation configuration must be a JSON object')
    scenes = config.get('test_paths', [])
    source_pairs = config.get('test_indices_src', [])
    target_pairs = config.get('test_indices_tgt', [])
    if (
        not isinstance(scenes, list) or not isinstance(source_pairs, list)
        or not isinstance(target_pairs, list) or not scenes or not source_pairs
        or len(source_pairs) != len(target_pairs)
    ):
        raise ValueError(
            f'{json_path}: provide non-empty test_paths and equally sized '
            'test_indices_src/test_indices_tgt lists'
        )
    pairs = []
    for pair_index, (source, target) in enumerate(zip(source_pairs, target_pairs)):
        for role, indices in (('src', source), ('tgt', target)):
            if not isinstance(indices, list) or not indices or any(
                type(index) is not int or index < 0 for index in indices
            ):
                raise ValueError(f'{json_path}: pair {pair_index} {role} must be a non-empty list of non-negative integers')
        pairs.append({'src': source, 'tgt': target})
    result = {}
    for scene in scenes:
        if not isinstance(scene, str) or not scene.strip():
            raise ValueError(f'{json_path}: test_paths must contain non-empty scene names or paths')
        name = Path(scene.rstrip('/')).name
        if not name or name in ('.', '..'):
            raise ValueError(f'{json_path}: invalid scene name {scene!r}')
        result[name] = pairs
    return result
