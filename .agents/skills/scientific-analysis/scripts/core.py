"""Deterministic local analysis helpers; no import-time I/O or network access."""
from __future__ import annotations

import hashlib
import json
import platform
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, KFold, TimeSeriesSplit

BRANDS = ['audi', 'bmw', 'ford', 'hyundi', 'merc', 'skoda', 'toyota', 'vauxhall', 'vw']
META = {'row_id', '_group', '_time'}


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(clean_json(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def safe_id(value: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}', value):
        raise ValueError('case_id and run_id must contain only letters, digits, underscores or hyphens')
    return value


def load_spec(path: Path) -> dict:
    spec = read_json(path)
    allowed = {'case_id', 'run_id', 'data_path', 'data_dir', 'task', 'target', 'group_column',
               'time_column', 'snapshot_year', 'sample_n', 'seed', 'folds', 'approved_features',
               'question', 'hypotheses', 'output_root', 'positive_class'}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f'Unknown spec fields: {sorted(unknown)}')
    for key in ['case_id', 'run_id', 'task']:
        if key not in spec:
            raise ValueError(f'Missing required field: {key}')
    safe_id(spec['case_id']); safe_id(spec['run_id'])
    if spec['task'] not in {'regression', 'classification', 'clustering', 'anomaly'}:
        raise ValueError('Unsupported task')
    if not isinstance(spec.get('approved_features', False), bool):
        raise ValueError('approved_features must be an explicit boolean')
    if spec['task'] in {'regression', 'classification'} and not spec.get('target'):
        raise ValueError('A supervised task requires target')
    if spec.get('folds', 3) < 2 or spec.get('sample_n', 12000) < 0:
        raise ValueError('folds must be >=2 and sample_n >=0')
    return spec


def load_data(spec: dict) -> tuple[pd.DataFrame, list[dict]]:
    records = []
    if spec.get('data_dir'):
        paths = [(Path(spec['data_dir']) / f'{name}.csv', name) for name in BRANDS]
    elif spec.get('data_path'):
        paths = [(Path(spec['data_path']), None)]
    else:
        raise ValueError('Specify data_path or the nine-brand data_dir')
    frames = []
    for path, brand in paths:
        if not path.is_file():
            raise FileNotFoundError(f'Missing source: {path}; no synthetic fallback is used')
        records.append({'path': str(path.resolve()), 'sha256': digest(path)})
        chunk = pd.read_csv(path).rename(columns={'tax(£)': 'tax'})
        if chunk.columns.duplicated().any() or META.intersection(chunk.columns):
            raise ValueError('Duplicate or reserved source columns')
        if brand:
            chunk['brand'] = brand
        for col in chunk.select_dtypes(include=['object', 'string']).columns:
            chunk[col] = chunk[col].astype('object').map(lambda x: x.strip() if isinstance(x, str) else x)
        chunk['row_id'] = [hashlib.sha256(f'{path.name}:{i}'.encode()).hexdigest()[:20] for i in range(len(chunk))]
        frames.append(chunk)
    frame = pd.concat(frames, ignore_index=True)
    if frame['row_id'].duplicated().any():
        raise ValueError('Source row identities collide')
    target = spec.get('target')
    if target and target not in frame:
        raise ValueError(f'Target column not found: {target}')
    if target and frame[target].isna().any():
        raise ValueError('Missing target values; define an explicit source exclusion policy')
    for col in frame.select_dtypes(include='number'):
        if np.isinf(frame[col].to_numpy(dtype=float)).any():
            raise ValueError(f'Infinite values in {col}')
    group = spec.get('group_column')
    if group:
        if group not in frame or frame[group].isna().any():
            raise ValueError('group_column missing or contains null values')
        frame['_group'] = frame[group].astype(str)
    else:
        original = [c for c in frame if c not in META and c != target]
        frame['_group'] = pd.util.hash_pandas_object(frame[original], index=False).astype(str)
    time = spec.get('time_column')
    if time:
        if time not in frame:
            raise ValueError('time_column missing')
        frame['_time'] = pd.to_datetime(frame[time], errors='raise', utc=True)
        if frame['_time'].isna().any() or frame['_time'].duplicated().any():
            raise ValueError('Time validation requires unique, non-null timestamps')
        frame = frame.sort_values('_time')
    limit = spec.get('sample_n', 12000)
    if limit and len(frame) > limit:
        if time:
            frame = frame.tail(limit)
        else:
            groups = frame['_group'].drop_duplicates().sample(frac=1, random_state=spec.get('seed', 42))
            sizes = frame.groupby('_group').size().reindex(groups)
            accepted = sizes.cumsum() <= limit
            frame = frame[frame['_group'].isin(groups[accepted.to_numpy()])]
    if len(frame) < 3:
        raise ValueError('Too few rows after group-preserving sampling; increase sample_n')
    return frame.reset_index(drop=True), records


def build_features(frame: pd.DataFrame, snapshot_year: int) -> pd.DataFrame:
    result = frame.copy()
    if not {'year', 'mileage'}.issubset(result):
        raise ValueError('Car feature set requires year and mileage')
    if (result['year'] > snapshot_year).any():
        raise ValueError('Vehicle year exceeds snapshot_year; review the snapshot contract')
    if (result['mileage'] < 0).any():
        raise ValueError('Negative mileage is outside the car data contract')
    result['vehicle_age'] = snapshot_year - result['year']
    result['log_mileage'] = np.log1p(result['mileage'])
    result['mileage_per_year'] = result['mileage'] / result['vehicle_age'].clip(lower=1)
    return result


def make_folds(frame: pd.DataFrame, task: str, folds: int, seed: int, target: str | None = None):
    if '_time' in frame:
        splits = list(TimeSeriesSplit(n_splits=folds).split(frame))
        for train, valid in splits:
            if not set(frame.iloc[train]['_group']).isdisjoint(frame.iloc[valid]['_group']):
                raise ValueError('Repeated groups cross time validation boundaries; supply a study-specific split')
        return splits
    if frame['_group'].duplicated().any():
        return list(GroupKFold(n_splits=folds).split(frame, groups=frame['_group']))
    if task == 'classification':
        return list(StratifiedKFold(folds, shuffle=True, random_state=seed).split(frame, frame[target]))
    return list(KFold(folds, shuffle=True, random_state=seed).split(frame))


def preprocessing(frame: pd.DataFrame, scale: bool = False):
    numeric = frame.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in frame if c not in numeric]
    steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(steps), numeric),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
                                  ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), categorical),
    ])


def validate_view(view: dict, manifest: dict, row_ids: set[str]) -> None:
    if view.get('run_id') != manifest['run_id'] or view.get('data_hash') != manifest['data_hash']:
        raise ValueError('View data version/run does not match this analysis')
    selected = view.get('selected_row_ids', view.get('selected_ids', []))
    if not isinstance(selected, list) or not set(selected).issubset(row_ids):
        raise ValueError('View contains unknown row identities')


def view_subset(view: dict, frame: pd.DataFrame) -> list[str]:
    filters = view.get('filters', {})
    if not isinstance(filters, dict) or set(filters) - {'numeric', 'category'}:
        raise ValueError('Invalid view filters')
    active = frame.copy()
    numeric = filters.get('numeric')
    if numeric is not None:
        if not isinstance(numeric, dict) or set(numeric) != {'column', 'min', 'max'}:
            raise ValueError('Invalid numeric filter')
        col, low, high = numeric['column'], numeric['min'], numeric['max']
        if col not in frame or not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError('Numeric filter column is unavailable or not numeric')
        for value in [low, high]:
            if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value)):
                raise ValueError('Invalid numeric filter bound')
        if low is not None and high is not None and low > high:
            raise ValueError('Numeric filter minimum exceeds maximum')
        active = active[active[col].notna()]
        if low is not None:
            active = active[active[col] >= low]
        if high is not None:
            active = active[active[col] <= high]
    category = filters.get('category')
    if category is not None:
        if not isinstance(category, dict) or set(category) != {'column', 'value'}:
            raise ValueError('Invalid category filter')
        col = category['column']
        if col not in frame or col in META or pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError('Category filter column is unavailable or not categorical')
        labels = active[col].map(lambda v: '（缺失）' if pd.isna(v) else str(v))
        if not isinstance(category['value'], str):
            raise ValueError('Category filter value must be a string')
        active = active[labels == category['value']]
    if 'filtered_row_ids' in view and (not isinstance(view['filtered_row_ids'], list) or
            set(view['filtered_row_ids']) != set(active['row_id'])):
        raise ValueError('Exported filtered row identities disagree with declared filters')
    selected = view.get('selected_row_ids', view.get('selected_ids', []))
    if selected:
        active = active[active['row_id'].isin(selected)]
    return active['row_id'].tolist()


def check_sources(manifest: dict):
    for source in manifest['sources']:
        if digest(Path(source['path'])) != source['sha256']:
            raise ValueError(f'Source changed since inspect: {source["path"]}')


def predictor_columns(frame: pd.DataFrame, spec: dict):
    excluded = META | {spec.get('target'), spec.get('group_column'), spec.get('time_column')}
    return [c for c in frame if c not in excluded]
