"""Library-backed training from memory-mapped features and external data pages."""
from contextlib import contextmanager
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import tempfile

from filelock import FileLock
import joblib
import lightgbm as lgb
import numpy as np
import psutil
from tqdm.auto import tqdm

from .disk_cache import MemoryBudgetError, atomic_json, safe_replace
from .features import FEATURE_COLUMNS

TRAIN_VERSION = 1


def model_config(config):
    """Index performance controls change neither model inputs nor fitted weights."""
    return {k: v for k, v in config.items() if k not in {'index_workers', 'index_cache_mb'}}


class DiskClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, kind, booster, device='cpu'):
        self.kind, self.booster, self.training_device_ = kind, booster, device

    def predict_proba(self, features):
        data = features[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        if self.kind == 'constant':
            values = np.full(len(data), self.booster, dtype=np.float32)
        elif self.kind == 'lightgbm':
            values = self.booster.predict(data)
        else:
            import xgboost as xgb
            values = self.booster.predict(xgb.DMatrix(data, feature_names=FEATURE_COLUMNS))
        return np.column_stack([1 - values, values])


def select_rows(matrix, fold, batch_rows):
    directory = matrix.directory / ('all' if fold is None else f'exclude-fold-{fold}')
    directory.mkdir(exist_ok=True)
    complete = directory / 'complete.json'
    with FileLock(str(directory / 'writer.lock')):
        if not complete.exists():
            count = positives = 0
            for start in range(0, len(matrix), batch_rows):
                mask = np.ones(min(batch_rows, len(matrix) - start), dtype=bool) if fold is None else matrix.fold[start:start+batch_rows] != fold
                count += int(mask.sum())
                positives += int(matrix.y[start:start+batch_rows][mask].sum())
            arrays = {name: np.lib.format.open_memmap(directory / f'{name}.npy', mode='w+', dtype=dtype, shape=(count,))
                      for name, dtype in [('indices', np.int64), ('labels', np.float32), ('weights', np.float32)]}
            cursor = 0
            for start in tqdm(range(0, len(matrix), batch_rows), desc=f'Select training rows (fold {fold})', leave=False):
                indices = np.arange(start, min(start+batch_rows, len(matrix)), dtype=np.int64)
                if fold is not None:
                    indices = indices[matrix.fold[indices] != fold]
                end = cursor + len(indices)
                arrays['indices'][cursor:end] = indices
                arrays['labels'][cursor:end] = matrix.y[indices]
                arrays['weights'][cursor:end] = matrix.weight[indices]
                cursor = end
            for array in arrays.values():
                array.flush()
            del arrays
            atomic_json(complete, {'rows': count, 'positives': positives})
    metadata = json.loads(complete.read_text())
    return directory, metadata


class MappedSequence(lgb.Sequence):
    def __init__(self, matrix, indices, batch_size, progress=None):
        self.matrix, self.indices, self.batch_size, self.progress = matrix, indices, batch_size, progress

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        result = np.asarray(self.matrix[self.indices[index]], dtype=np.float64)
        if isinstance(index, slice) and self.progress is not None:
            self.progress.update(len(result))
        return result


def _memory_preflight(kind, rows, config, budget_gb):
    gc.collect()
    # Labels, gradients and (for LightGBM) compressed bins remain resident.
    # Feature/raw text matrices are not part of this heap allocation.
    estimate = int(rows * ((len(FEATURE_COLUMNS) + 40) * 1.5 if kind == 'lightgbm' else 40) + 256 * 2**20)
    budget = int(budget_gb * 2**30) if budget_gb else int(psutil.virtual_memory().available * .7)
    result = {'estimated_resident_bytes': estimate, 'budget_bytes': budget, 'rows': rows}
    if estimate > budget:
        raise MemoryBudgetError(f'{kind}: feature data is on disk, but compressed bins/labels/gradients still need approximately '
                               f'{estimate/2**30:.2f} GiB, above the {budget/2**30:.2f} GiB budget. '
                               'All preprocessing/features are checkpointed. Use --train-query-limit for a bounded training subset.')
    if kind == 'xgboost' and config.get('device', 'cpu').startswith('cuda'):
        ordinal = config['device'].split(':')[-1] if ':' in config['device'] else '0'
        status = subprocess.run(['nvidia-smi', '-i', ordinal, '--query-gpu=memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        if status.returncode:
            raise RuntimeError('CUDA requested but GPU memory could not be queried; refusing silent CPU fallback')
        free = float(status.stdout.strip()) * 2**20
        required = rows * 24 + config.get('training_batch_rows', 100_000) * len(FEATURE_COLUMNS) * 12 + 256 * 2**20
        result.update(estimated_gpu_bytes=required, available_gpu_bytes=free)
        if required > free * .8:
            raise MemoryBudgetError('XGBoost external pages are on disk, but per-row GPU state exceeds the conservative VRAM budget. '
                                   'Use --train-query-limit or explicitly configure device=cpu; no CPU fallback was performed.')
    return result


def fit_disk(kind, matrix, config, fold=None, budget_gb=None):
    batch_rows = config.get('training_batch_rows', 100_000)
    selection, metadata = select_rows(matrix, fold, batch_rows)
    if not metadata['rows']:
        raise ValueError('No training candidates in this fold')
    key = hashlib.sha256(json.dumps([TRAIN_VERSION, kind, model_config(config), version(kind)], sort_keys=True).encode()).hexdigest()[:20]
    directory = selection / f'model-{key}'
    directory.mkdir(exist_ok=True)
    model_path = directory / 'model.joblib'
    with FileLock(str(directory / 'writer.lock')):
        if model_path.exists():
            tqdm.write(f'MODEL CACHE HIT {kind}, held-out fold {fold}')
            return joblib.load(model_path)
        memory = _memory_preflight(kind, metadata['rows'], config, budget_gb)
        atomic_json(directory / 'memory.json', memory)
        if metadata['positives'] in [0, metadata['rows']]:
            model = DiskClassifier('constant', int(metadata['positives'] > 0))
        else:
            indices = np.load(selection / 'indices.npy', mmap_mode='r')
            labels = np.load(selection / 'labels.npy', mmap_mode='r')
            weights = np.load(selection / 'weights.npy', mmap_mode='r')
            with tqdm(total=config['n_estimators'], desc=f'{kind} disk training (fold {fold})') as bar:
                if kind == 'lightgbm':
                    params = {'objective': 'binary', 'num_leaves': config['num_leaves'],
                                         'learning_rate': config['learning_rate'], 'seed': config['seed'],
                                         'num_threads': config['threads'], 'verbosity': -1, 'deterministic': True,
                                         'force_col_wise': True, 'min_data_in_leaf': 10,
                                         'max_bin': config.get('max_bin', 64)}
                    with tqdm(total=len(indices), desc='LightGBM bins from disk', unit='row', leave=False) as read_bar:
                        data = lgb.Dataset(MappedSequence(matrix.x, indices, batch_rows, read_bar), label=labels,
                                           weight=weights, feature_name=FEATURE_COLUMNS, free_raw_data=True, params=params)
                        data.construct()
                    booster = lgb.train(params, data,
                                        num_boost_round=config['n_estimators'], callbacks=[lambda env: bar.update(1)])
                    booster.free_dataset()
                    del data
                    model = DiskClassifier(kind, booster)
                elif kind == 'xgboost':
                    import xgboost as xgb

                    class Iterator(xgb.DataIter):
                        def __init__(self, prefix):
                            self.cursor = 0
                            # DMatrix external pages accept CPU NumPy batches and
                            # this installed Windows build can train them on CUDA.
                            super().__init__(cache_prefix=prefix, on_host=False, release_data=True)

                        def reset(self):
                            self.cursor = 0
                            page_bar.reset()

                        def next(self, input_data):
                            if self.cursor >= len(indices):
                                return False
                            end = min(len(indices), self.cursor + batch_rows)
                            input_data(data=np.asarray(matrix.x[indices[self.cursor:end]], dtype=np.float32),
                                       label=labels[self.cursor:end], weight=weights[self.cursor:end], feature_names=FEATURE_COLUMNS)
                            page_bar.update(end-self.cursor)
                            self.cursor = end
                            return True

                    class Progress(xgb.callback.TrainingCallback):
                        def after_iteration(self, model, epoch, evals_log):
                            bar.update(1)
                            return False

                    with tempfile.TemporaryDirectory(dir=directory, prefix='pages-', ignore_cleanup_errors=True) as temporary:
                        with tqdm(total=len(indices), desc='XGBoost pages to disk', unit='row', leave=False) as page_bar:
                            iterator = Iterator(str(Path(temporary) / 'cache'))
                            data = xgb.DMatrix(iterator, nthread=config['threads'])
                        booster = xgb.train({'objective': 'binary:logistic', 'tree_method': 'hist',
                                             'device': config.get('device', 'cpu'), 'max_bin': config.get('max_bin', 64),
                                             'max_depth': config.get('disk_max_depth', 5), 'max_leaves': config['num_leaves'],
                                             'eta': config['learning_rate'], 'seed': config['seed'], 'nthread': config['threads']},
                                            data, num_boost_round=config['n_estimators'], callbacks=[Progress()])
                        actual = json.loads(booster.save_config())['learner']['generic_param']['device']
                        if config.get('device', 'cpu').startswith('cuda') and not actual.startswith('cuda'):
                            raise RuntimeError('XGBoost silently fell back from CUDA; refusing this result')
                        tqdm.write(f'External-memory XGBoost training device: {actual}')
                        model = DiskClassifier(kind, booster, actual)
                        del data, iterator
                        gc.collect()
                else:
                    raise ValueError(f'Unsupported disk model: {kind}')
        joblib.dump(model, directory / 'model.tmp')
        safe_replace(directory / 'model.tmp', model_path)
        atomic_json(directory / 'complete.json', {'kind': kind, 'fold': fold, 'training_device': model.training_device_, **metadata})
    gc.collect()
    return model
