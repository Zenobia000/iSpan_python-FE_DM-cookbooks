#!/usr/bin/env python3
"""Local evidence-producing CLI for the scientific-analysis skill."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from core import (META, build_features, check_sources, clean_json, load_data, load_spec,
                  make_folds, object_hash, predictor_columns, preprocessing, read_json,
                  safe_id, validate_view, view_subset, write_json, digest)


def inspect(spec, folder):
    frame, sources = load_data(spec)
    splits = []
    if spec['task'] in {'regression', 'classification'}:
        splits = [{'fold_id': k, 'train': t.tolist(), 'valid': v.tolist()} for k, (t, v) in enumerate(
            make_folds(frame, spec['task'], spec.get('folds', 3), spec.get('seed', 42), spec.get('target')))]
    folder.mkdir(parents=True, exist_ok=False)
    frame.to_csv(folder / 'data.csv', index=False)
    write_json(folder / 'splits.json', splits)
    manifest = {'schema_version': 1, 'run_id': spec['run_id'], 'case_id': spec['case_id'],
                'working_data_hash': digest(folder / 'data.csv'),
                'splits_hash': digest(folder / 'splits.json'),
                'data_hash': object_hash(sources), 'sources': sources, 'spec_hash': object_hash(spec),
                'row_count': len(frame), 'spec': spec, 'validation': 'development CV only; no external validation',
                'versions': {p: importlib.metadata.version(p) for p in ['pandas', 'numpy', 'scikit-learn']}}
    write_json(folder / 'manifest.json', manifest)
    write_json(folder / 'research.json', {'question': spec.get('question', ''),
               'hypotheses': spec.get('hypotheses', []), 'decisions': [],
               'limitations': ['Observational associations do not establish causality.',
                               'Proxy groups are not verified patient/vehicle identities.']})


def explore(spec, folder, frame):
    profile = {'rows': len(frame), 'columns': {}, 'correlations': {}}
    if (folder / 'view.json').exists():
        view = read_json(folder / 'view.json')
        selected = view['effective_row_ids']
        profile['expert_selected_rows'] = len(selected)
        profile['follow_up_status'] = 'Post-exploration subgroup; no independent confirmation'
        subset = frame[frame['row_id'].isin(selected)]
        profile['selected_numeric_summary'] = subset.select_dtypes(include='number').describe().to_dict()
        profile['empty_selection'] = not selected
    for col in predictor_columns(frame, spec) + ([spec['target']] if spec.get('target') else []):
        values = frame[col]
        entry = {'missing': int(values.isna().sum()), 'unique': int(values.nunique())}
        if pd.api.types.is_numeric_dtype(values):
            entry.update({'median': values.median(), 'skew': values.skew(), 'excess_kurtosis': values.kurt(),
                          'quantiles': values.quantile([0, .25, .75, 1]).to_dict()})
        profile['columns'][col] = entry
    profile['correlations'] = frame.drop(columns=list(META), errors='ignore').select_dtypes(include='number').corr().to_dict()
    write_json(folder / 'profile.json', profile)


def features(spec, folder, frame):
    if not spec.get('approved_features', False):
        raise ValueError('Feature hypotheses require explicit approved_features=true before transformation')
    enhanced = build_features(frame, spec.get('snapshot_year', 2020))
    output = folder / 'features.csv'
    if output.exists():
        raise FileExistsError(output)
    enhanced.to_csv(output, index=False)
    write_json(folder / 'feature_dictionary.json', {
        'vehicle_age': {'formula': 'snapshot_year - year', 'unit': 'year', 'fit_scope': 'row'},
        'log_mileage': {'formula': 'log1p(mileage)', 'unit': 'transformed miles', 'fit_scope': 'row'},
        'mileage_per_year': {'formula': 'mileage / max(vehicle_age, 1)', 'unit': 'miles/year', 'fit_scope': 'row',
                             'limitation': 'New-car age denominator approximated as one year'},
    })
    write_json(folder / 'feature_manifest.json', {'sha256': digest(output)})


def model(spec, folder, frame):
    from sklearn.dummy import DummyRegressor, DummyClassifier
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.tree import export_text
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, balanced_accuracy_score, log_loss
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMRegressor, LGBMClassifier
    if (folder / 'metrics.json').exists():
        raise FileExistsError(folder / 'metrics.json')
    if spec['task'] not in {'regression', 'classification'}:
        raise ValueError('MVP supports supervised model evaluation; clustering/anomaly are not implemented yet')
    seed = spec.get('seed', 42)
    classification = spec['task'] == 'classification'
    if classification:
        factories = {'dummy': lambda: DummyClassifier(strategy='prior'),
                     'linear': lambda: LogisticRegression(max_iter=1500, random_state=seed),
                     'shallow_tree': lambda: DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=seed),
                     'lightgbm': lambda: LGBMClassifier(n_estimators=100, max_depth=5, num_leaves=15, n_jobs=2, random_state=seed, verbosity=-1)}
    else:
        factories = {'dummy': lambda: DummyRegressor(strategy='median'), 'linear': lambda: Ridge(alpha=1.),
                     'shallow_tree': lambda: DecisionTreeRegressor(max_depth=3, min_samples_leaf=5, random_state=seed),
                     'lightgbm': lambda: LGBMRegressor(n_estimators=100, max_depth=5, num_leaves=15, n_jobs=2, random_state=seed, verbosity=-1)}
    variants = {'baseline': frame}
    if (folder / 'features.csv').exists():
        if digest(folder / 'features.csv') != read_json(folder / 'feature_manifest.json')['sha256']:
            raise ValueError('Feature artifact changed since generation')
        variants['features_v1'] = pd.read_csv(folder / 'features.csv', dtype={'row_id': str, '_group': str})
    splits = read_json(folder / 'splits.json')
    experiments, predictions, interpretations = [], [], []
    for variant, data in variants.items():
        if data['row_id'].tolist() != frame['row_id'].tolist():
            raise ValueError('Feature rows are not paired with baseline')
        columns = predictor_columns(data, spec)
        X, y = data[columns], data[spec['target']]
        if classification and y.nunique() < 2:
            raise ValueError('Classification needs at least two classes')
        for name, factory in factories.items():
            scores = []
            for split in splits:
                train, valid = split['train'], split['valid']
                if classification and y.iloc[train].nunique() != y.nunique():
                    raise ValueError('Training fold lacks classes; revise the validation design')
                estimator = Pipeline([('preprocessing', preprocessing(X.iloc[train], scale=name == 'linear')),
                                      ('model', factory())])
                estimator.fit(X.iloc[train], y.iloc[train])
                if name in {'linear', 'shallow_tree'}:
                    fitted = estimator.named_steps['model']
                    names = estimator.named_steps['preprocessing'].get_feature_names_out().tolist()
                    record = {'model': name, 'variant': variant, 'fold_id': split['fold_id'],
                              'limitation': 'Fold-specific model association, not causal effects; numeric linear coefficients use training-fold standardized inputs; one-hot coefficients depend on coding and collinearity.'}
                    if name == 'linear':
                        weights = np.atleast_2d(fitted.coef_)
                        record['coefficients'] = [{'class_index': k, 'total_terms': len(names), 'terms': sorted(
                            [{'feature': feature, 'coefficient': float(coefficient)} for feature, coefficient in zip(names, row)],
                            key=lambda term: abs(term['coefficient']), reverse=True)} for k, row in enumerate(weights)]
                        record['classes'] = fitted.classes_.tolist() if classification else None
                        record['intercept'] = np.atleast_1d(fitted.intercept_).tolist()
                    else:
                        record['rules'] = export_text(fitted, feature_names=names, max_depth=3)
                    interpretations.append(record)
                predicted = estimator.predict(X.iloc[valid])
                if classification:
                    values = {'accuracy': accuracy_score(y.iloc[valid], predicted),
                              'balanced_accuracy': balanced_accuracy_score(y.iloc[valid], predicted),
                              'log_loss': log_loss(y.iloc[valid], estimator.predict_proba(X.iloc[valid]), labels=estimator.classes_)}
                else:
                    values = {'mae': mean_absolute_error(y.iloc[valid], predicted),
                              'rmse': float(np.sqrt(mean_squared_error(y.iloc[valid], predicted))),
                              'r2': r2_score(y.iloc[valid], predicted)}
                scores.append({'fold_id': split['fold_id'], 'n_train': len(train), 'n_valid': len(valid), **values})
                for pos, prediction in zip(valid, predicted):
                    predictions.append({'row_id': frame.iloc[pos]['row_id'], 'model': name, 'variant': variant,
                                        'fold_id': split['fold_id'], 'y_true': y.iloc[pos], 'y_pred': prediction})
            aggregate = {metric: {'mean': np.mean([s[metric] for s in scores]), 'std': np.std([s[metric] for s in scores], ddof=1)} for metric in values}
            experiments.append({'model': name, 'variant': variant, 'folds': scores, 'aggregate': aggregate,
                                'params': factory().get_params()})
    comparisons = []
    main = 'log_loss' if classification else 'mae'
    for exp in experiments:
        if exp['variant'] == 'features_v1':
            baseline = next(e for e in experiments if e['model'] == exp['model'] and e['variant'] == 'baseline')
            comparisons.append({'model': exp['model'], 'metric': main, 'positive_means': 'improved',
                                'fold_improvement': [a[main] - b[main] for a, b in zip(baseline['folds'], exp['folds'])]})
    pd.DataFrame(predictions).to_csv(folder / 'oof.csv', index=False)
    write_json(folder / 'model_interpretations.json', interpretations)
    write_json(folder / 'metrics.json', {'schema_version': 1, 'task': spec['task'], 'primary_metric': main,
               'validation': 'development cross-validation; exploratory model comparison, not independent confirmation',
               'experiments': experiments, 'paired_comparisons': comparisons,
               'selection': 'Human comparison required; no automatic replacement or causal claim'})


def explain(spec, folder, frame):
    if not (folder / 'oof.csv').exists():
        raise ValueError('Run model first')
    oof = pd.read_csv(folder / 'oof.csv', dtype={'row_id': str})
    output = {'method': 'OOF error diagnosis; SHAP and final-model attribution are outside MVP', 'groups': []}
    output['model_interpretations'] = read_json(folder / 'model_interpretations.json')
    if spec['task'] == 'regression':
        oof['absolute_error'] = abs(oof['y_true'] - oof['y_pred'])
        group = 'brand' if 'brand' in frame else '_group'
        joined = oof.merge(frame[['row_id', group]], on='row_id', validate='many_to_one')
        output['groups'] = joined.groupby(['model', 'variant', group])['absolute_error'].agg(['mean', 'count']).reset_index().to_dict('records')
    write_json(folder / 'explanation.json', output)


def report(spec, folder, frame):
    from reporting import render_report
    output = folder / 'report.html'
    if output.exists():
        raise FileExistsError(output)
    manifest = read_json(folder / 'manifest.json')
    if (folder / 'features.csv').exists():
        if digest(folder / 'features.csv') != read_json(folder / 'feature_manifest.json')['sha256']:
            raise ValueError('Feature artifact changed since generation')
        frame = pd.read_csv(folder / 'features.csv', dtype={'row_id': str, '_group': str})
    data = frame.drop(columns=['_group', '_time'], errors='ignore').copy()
    metrics = read_json(folder / 'metrics.json') if (folder / 'metrics.json').exists() else {}
    if (folder / 'oof.csv').exists() and spec['task'] == 'regression':
        oof = pd.read_csv(folder / 'oof.csv', dtype={'row_id': str})
        selected = oof[(oof['model'] == 'linear') & (oof['variant'] == 'baseline')][['row_id', 'y_pred']]
        data = data.merge(selected, on='row_id', how='left', validate='one_to_one')
        data['residual'] = data[spec['target']] - data['y_pred']
    findings = [{'observation': spec.get('question', 'Inspect distributions and compare plausible explanations.'),
                 'alternative': 'Batch, selection and measurement differences can produce the same pattern.',
                 'limitation': 'Associations and predictive scores do not establish a root cause.',
                 'next_step': 'Select a subgroup, annotate the concern and return the exported view to the agent.'}]
    render_report(clean_json({'schema_version': 1, 'rows': data.to_dict('records'),
                  'numeric_columns': data.select_dtypes(include='number').columns.tolist(),
                  'target': spec.get('target'), 'findings': findings, 'metrics': metrics,
                  'interpretations': read_json(folder / 'model_interpretations.json') if (folder / 'model_interpretations.json').exists() else [],
                  'resume_state': read_json(folder / 'view.json') if (folder / 'view.json').exists() else None,
                  'run_id': manifest['run_id'], 'data_hash': manifest['data_hash'],
                  'column_units': {'price': 'GBP', 'mileage': 'miles', 'year': 'year', 'engineSize': 'L',
                                   'vehicle_age': 'years', 'log_mileage': 'log1p(miles)',
                                   'tax': 'GBP (period unspecified)', 'mpg': 'mpg (gallon convention unverified)',
                                   'y_pred': 'GBP', 'residual': 'GBP',
                                   'mileage_per_year': 'miles/year'} if spec.get('data_dir') else {},
                  'question': spec.get('question', ''), 'parent_run_id': manifest.get('parent_run_id')}), output)
    summary = '# Research evidence summary\n\n' + spec.get('question', '') + '\n\n'
    summary += 'Development CV only. No independent confirmation or causal identification is claimed.\n\n'
    for exp in metrics.get('experiments', []):
        summary += f'- {exp["model"]} / {exp["variant"]}: {json.dumps(clean_json(exp["aggregate"]))}\n'
    with (folder / 'summary.md').open('x', encoding='utf-8') as stream:
        stream.write(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inspect', 'explore', 'features', 'model', 'explain', 'report', 'resume'])
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--view', type=Path)
    parser.add_argument('--new-run-id')
    args = parser.parse_args(argv)
    spec = load_spec(args.spec)
    folder = Path(spec.get('output_root', 'workspace')) / safe_id(spec['case_id']) / safe_id(spec['run_id'])
    if args.stage == 'inspect':
        inspect(spec, folder)
    else:
        manifest = read_json(folder / 'manifest.json')
        if manifest['spec_hash'] != object_hash(spec):
            raise ValueError('Spec changed; create a new run instead of changing existing evidence')
        check_sources(manifest)
        if digest(folder / 'data.csv') != manifest['working_data_hash']:
            raise ValueError('Working data changed since inspect; create a new analysis')
        if digest(folder / 'splits.json') != manifest['splits_hash']:
            raise ValueError('Validation splits changed since inspect')
        frame = pd.read_csv(folder / 'data.csv', dtype={'row_id': str, '_group': str})
        if args.stage == 'resume':
            if not args.view or not args.new_run_id:
                raise ValueError('resume requires --view and --new-run-id')
            view = read_json(args.view)
            validate_view(view, manifest, set(frame['row_id']))
            analysis_frame = frame
            if (folder / 'features.csv').exists():
                if digest(folder / 'features.csv') != read_json(folder / 'feature_manifest.json')['sha256']:
                    raise ValueError('Feature artifact changed since generation')
                analysis_frame = pd.read_csv(folder / 'features.csv', dtype={'row_id': str, '_group': str})
            view['effective_row_ids'] = view_subset(view, analysis_frame)
            new_spec = {**spec, 'run_id': safe_id(args.new_run_id)}
            target = folder.parent / new_spec['run_id']
            target.mkdir(parents=True, exist_ok=False)
            frame.to_csv(target / 'data.csv', index=False)
            write_json(target / 'manifest.json', {**manifest, 'run_id': new_spec['run_id'],
                       'working_data_hash': digest(target / 'data.csv'),
                       'parent_run_id': spec['run_id'], 'spec': new_spec, 'spec_hash': object_hash(new_spec)})
            write_json(target / 'spec.json', new_spec)
            write_json(target / 'splits.json', read_json(folder / 'splits.json'))
            write_json(target / 'research.json', {**read_json(folder / 'research.json'), 'expert_view': view,
                       'follow_up': 'Exploratory subgroup selected after viewing evidence; not independent validation'})
            write_json(target / 'view.json', view)
            import shutil
            for name in ['features.csv', 'feature_dictionary.json', 'feature_manifest.json']:
                if (folder / name).exists():
                    shutil.copyfile(folder / name, target / name)
            print(f'Created research branch: {target}; use its spec.json for follow-up commands')
        else:
            globals()[args.stage](spec, folder, frame)
    print(f'{args.stage}: {folder}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, KeyError) as error:
        print(f'Analysis stopped: {error}', file=sys.stderr)
        sys.exit(2)
