import hashlib
import inspect
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'topppi-shared-backbone-v1'

def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def _read_torch(path):
    import torch
    options = {'map_location': 'cpu', 'weights_only': True}
    if 'mmap' in inspect.signature(torch.load).parameters:
        options['mmap'] = True
    try:
        return torch.load(path, **options)
    except RuntimeError as error:
        if 'mmap can only be used' not in str(error):
            raise
        options.pop('mmap', None)
        return torch.load(path, **options)

@lru_cache(maxsize=1)
def _shared_state(base_dir, expected_manifest_hash):
    directory = Path(base_dir)
    manifest_path = directory / 'manifest.json'
    if file_sha256(manifest_path) != expected_manifest_hash:
        raise ValueError('Shared ESM-2 manifest does not match the trained checkpoint.')
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    state = {}
    for shard in manifest['shards']:
        path = directory / shard['path']
        if path.parent.resolve() != directory.resolve():
            raise ValueError('Invalid backbone shard path.')
        if path.stat().st_size != shard['size'] or file_sha256(path) != shard['sha256']:
            raise ValueError(f'Incomplete or changed backbone shard: {path}')
        values = _read_torch(path)
        if state.keys() & values.keys():
            raise ValueError('Duplicate keys in shared backbone shards.')
        state.update(values)
    return state

def read_checkpoint_state(path, expected_task=None):

    payload = _read_torch(Path(path))
    if isinstance(payload, dict) and payload.get('format') == FORMAT:
        if expected_task is not None and payload['task'] != expected_task:
            raise ValueError(f"Checkpoint task {payload['task']} does not match {expected_task}.")
        base_dir = (ROOT / payload['base_dir']).resolve()
        if not base_dir.is_relative_to(ROOT):
            raise ValueError('Backbone directory must be inside this repository.')
        base = _shared_state(str(base_dir), payload['base_manifest_sha256'])
        state = dict(payload['state_dict'])
        mapping = payload['base_mapping']
        if state.keys() & mapping.keys():
            raise ValueError('A tensor is present in both checkpoint sections.')
        state.update({key: base[base_key] for key, base_key in mapping.items()})
        if set(state) != set(payload['all_original_keys']):
            raise ValueError('Reconstructed checkpoint key set is incomplete.')
        return state
    state = payload
    if isinstance(state, dict):
        for name in ('model_state_dict', 'model_state', 'state_dict'):
            if name in state:
                state = state[name]
                break
    if not isinstance(state, dict) or not state or not all(hasattr(v, 'shape') for v in state.values()):
        raise ValueError(f'No model state_dict found in {path}')
    return {key.removeprefix('module.'): value for key, value in state.items()}
