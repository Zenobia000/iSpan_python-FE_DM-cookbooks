import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / '.agents/skills/scientific-analysis/scripts/core.py'
spec = importlib.util.spec_from_file_location('harness_core', SCRIPT)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def test_features_do_not_use_target_or_future_year():
    frame = pd.DataFrame({'year': [2020, 2018], 'mileage': [0, 20000], 'price': [10, 20]})
    result = core.build_features(frame, 2020)
    assert result['mileage_per_year'].tolist() == [0, 10000]
    changed = frame.assign(price=[999, 888])
    pd.testing.assert_frame_equal(result.drop(columns='price'), core.build_features(changed, 2020).drop(columns='price'))
    with pytest.raises(ValueError, match='snapshot'):
        core.build_features(frame.assign(year=2021), 2020)


def test_groups_are_isolated_and_rows_are_paired():
    frame = pd.DataFrame({'row_id': list(map(str, range(30))), '_group': np.repeat(np.arange(10), 3)})
    folds = core.make_folds(frame, 'regression', 3, 42)
    for train, valid in folds:
        assert set(frame.iloc[train]['_group']).isdisjoint(frame.iloc[valid]['_group'])
        assert not set(train).intersection(valid)
    assert sorted(np.concatenate([v for _, v in folds]).tolist()) == list(range(30))


def test_group_sampling_preserves_complete_groups(tmp_path):
    source = tmp_path / 'sample.csv'
    pd.DataFrame({'group': np.repeat(['a', 'b', 'c', 'd', 'e'], 3), 'x': np.arange(15), 'target': np.arange(15)}).to_csv(source, index=False)
    frame, _ = core.load_data({'data_path': str(source), 'target': 'target', 'group_column': 'group', 'sample_n': 8})
    assert len(frame) == 6
    assert frame.groupby('_group').size().eq(3).all()


def test_imputer_only_learns_training_values():
    train = pd.DataFrame({'x': [1., 3., np.nan], 'category': ['a', 'a', 'b']})
    pipeline = core.preprocessing(train)
    pipeline.fit(train)
    a = pipeline.transform(pd.DataFrame({'x': [np.nan], 'category': ['unknown']}))
    assert a[0, 0] == 2
    assert pipeline.transform(pd.DataFrame({'x': [100000.], 'category': ['unknown']})).shape == a.shape


def test_resume_rejects_wrong_data_and_unknown_rows(tmp_path):
    manifest = {'run_id': 'first', 'data_hash': 'abc'}
    view = {'run_id': 'first', 'data_hash': 'wrong', 'selected_ids': ['a']}
    with pytest.raises(ValueError, match='version'):
        core.validate_view(view, manifest, {'a'})
    view['data_hash'] = 'abc'
    view['selected_ids'] = ['missing']
    with pytest.raises(ValueError, match='row'):
        core.validate_view(view, manifest, {'a'})


def test_view_filters_are_evaluated_not_trusted():
    frame = pd.DataFrame({'row_id': ['a', 'b', 'c'], 'x': [1, 2, 3], 'batch': ['A', 'A', 'B']})
    view = {'filters': {'numeric': {'column': 'x', 'min': 2, 'max': None}, 'category': {'column': 'batch', 'value': 'A'}}, 'selected_row_ids': []}
    assert core.view_subset(view, frame) == ['b']
    view['filtered_row_ids'] = ['a']
    with pytest.raises(ValueError, match='disagree'):
        core.view_subset(view, frame)
    del view['filtered_row_ids']
    view['filters']['numeric']['min'] = '2'
    with pytest.raises(ValueError, match='bound'):
        core.view_subset(view, frame)


def test_json_rejects_nonfinite_and_stage_overwrite(tmp_path):
    core.write_json(tmp_path / 'metrics.json', {'undefined': float('nan')})
    assert json.loads((tmp_path / 'metrics.json').read_text()) == {'undefined': None}
    with pytest.raises(FileExistsError):
        core.write_json(tmp_path / 'metrics.json', {})


def test_cli_real_workflow_raw_preserved_and_resume(tmp_path):
    source = tmp_path / 'source.csv'
    pd.DataFrame({'year': np.tile([2016, 2017, 2018], 12), 'mileage': np.arange(36) * 1000,
                  'fuel': np.tile(['petrol', 'diesel'], 18), 'price': 25000 - np.arange(36) * 200}).to_csv(source, index=False)
    original = source.read_bytes()
    settings = {'case_id': 'fixture', 'run_id': 'first', 'data_path': str(source), 'task': 'regression',
                'target': 'price', 'approved_features': True, 'output_root': str(tmp_path / 'runs')}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(settings))
    entry = SCRIPT.parent / 'run.py'
    def run(stage, *extra):
        return subprocess.run([sys.executable, str(entry), stage, '--spec', str(path), *extra], capture_output=True, text=True)
    for stage in ['inspect', 'explore', 'features', 'model', 'explain', 'report']:
        completed = run(stage)
        assert completed.returncode == 0, completed.stderr
    folder = tmp_path / 'runs/fixture/first'
    assert source.read_bytes() == original
    assert 'cross-validation' in json.loads((folder / 'metrics.json').read_text())['validation']
    assert (folder / 'report.html').stat().st_size > 1000000
    assert run('inspect').returncode == 2
    oof = pd.read_csv(folder / 'oof.csv')
    assert len(oof) == 36 * 4 * 2
    manifest = json.loads((folder / 'manifest.json').read_text())
    view = tmp_path / 'view.json'
    view.write_text(json.dumps({'run_id': 'first', 'data_hash': manifest['data_hash'],
                               'selected_row_ids': [], 'filters': {'numeric': {'column': 'mileage', 'min': 10000, 'max': 15000}},
                               'notes': 'Check this subgroup'}))
    assert run('resume', '--view', str(view), '--new-run-id', 'second').returncode == 0
    follow_up = tmp_path / 'runs/fixture/second'
    assert json.loads((follow_up / 'research.json').read_text())['expert_view']['notes'] == 'Check this subgroup'
    follow_spec = follow_up / 'spec.json'
    completed = subprocess.run([sys.executable, str(entry), 'explore', '--spec', str(follow_spec)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((follow_up / 'profile.json').read_text())['expert_selected_rows'] == 6
    assert (folder / 'report.html').is_file()
    (folder / 'splits.json').write_text('[]')
    assert 'splits changed' in run('report').stderr
