#!/usr/bin/env python3
"""Deterministic, standard-library-only model promotion gate."""

import argparse
import json
import sys
import hashlib
import random
import os
import shutil
import tempfile


GATE_CONFIG = {
    'primary_metric': 'heldout_exact',
    'bootstrap_rounds': 10000,
    'ci': 0.95,
    'comparators': ['frozen_0_8b', 'unadapted_base'],
    'regression_required': 1.0,
    'critical_fabrications_allowed': 0,
    'audit_min_blind_sample': 200,
    'seed': 20261120,
}


def sha256_hex(b) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json_bytes(obj) -> bytes:
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=True,
    ).encode('utf-8')


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description='Vet run artifacts and render deterministic promotion verdicts.'
    )
    parser.add_argument(
        '--dump-config',
        action='store_true',
        help='print the gate configuration as JSON and exit',
    )
    subparsers = parser.add_subparsers(dest='command')

    vet_parser = subparsers.add_parser('vet', help='validate a run directory')
    vet_parser.add_argument('rundir')
    vet_parser.add_argument('--run-tag', default='untagged')

    verdict_parser = subparsers.add_parser(
        'verdict', help='evaluate a run directory and emit a verdict'
    )
    verdict_parser.add_argument('rundir')
    verdict_parser.add_argument('--run-tag', default='untagged')

    selftest_parser = subparsers.add_parser(
        'selftest', help='run deterministic synthetic control tests'
    )
    selftest_parser.add_argument('--run-tag', default='untagged')

    return parser


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.dump_config:
        print(json.dumps(GATE_CONFIG, sort_keys=True))
        return 0

    try:
        if args.command == 'vet':
            return int(cmd_vet(args.rundir, getattr(args, 'run_tag', 'untagged')) or 0)
        elif args.command == 'verdict':
            return int(cmd_verdict(args.rundir, args.run_tag) or 0)
        elif args.command == 'selftest':
            return int(cmd_selftest(args.run_tag) or 0)
        else:
            raise ValueError('a command is required: vet, verdict, or selftest')
    except Exception as e:
        print('FAIL_CLOSED: ' + repr(e), file=sys.stderr)
        return 3





import json


class GateError(Exception):
    pass


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.loads(handle.read())
    except Exception as exc:
        raise GateError("BAD_JSON " + str(path)) from exc


def load_jsonl(path):
    try:
        records = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("JSONL record must be an object")
                records.append(record)
        return records
    except Exception as exc:
        raise GateError("BAD_JSONL " + str(path)) from exc


import os


def load_rundir(rundir, cfg=None):
    if cfg is None:
        cfg = GATE_CONFIG

    candidates_doc = load_json(os.path.join(rundir, 'candidates.json'))
    if isinstance(candidates_doc, dict):
        if 'primary_index' in candidates_doc and candidates_doc['primary_index'] != 0:
            raise GateError('POST_HOC_SELECTION')
        candidates = candidates_doc.get('candidates')
    elif isinstance(candidates_doc, list):
        candidates = candidates_doc
    else:
        raise GateError('BAD_CANDIDATES')
    if not isinstance(candidates, list) or not candidates:
        raise GateError('BAD_CANDIDATES')

    primary = candidates[0]
    required = ('model_id', 'metrics_file', 'peritem_file', 'audit_file')
    if not isinstance(primary, dict) or any(key not in primary for key in required):
        raise GateError('BAD_PRIMARY')

    metrics = load_json(os.path.join(rundir, primary['metrics_file']))
    if (not isinstance(metrics, dict)
            or metrics.get('primary_metric') != cfg['primary_metric']):
        raise GateError('METRIC_MISMATCH')

    def read_scores(path):
        scores = {}
        for row in load_jsonl(path):
            if (not isinstance(row, dict) or 'item_id' not in row
                    or 'score' not in row):
                raise GateError('BAD_PERITEM_ROW')
            scores[row['item_id']] = row['score']
        return scores

    cand_scores = read_scores(os.path.join(rundir, primary['peritem_file']))
    audit = load_jsonl(os.path.join(rundir, primary['audit_file']))
    for row in audit:
        if (not isinstance(row, dict) or 'item_id' not in row
                or row.get('checker') not in ('deterministic', 'reviewer')):
            raise GateError('BAD_AUDIT_ROW')
        item_id = row['item_id']
        if item_id not in cand_scores:
            raise GateError('AUDIT_UNKNOWN_ITEM ' + str(item_id))

    comp_scores = {}
    for comp in cfg['comparators']:
        comp_dir = os.path.join(rundir, 'baseline', comp)
        scores = read_scores(os.path.join(comp_dir, 'peritem.jsonl'))
        if set(scores) != set(cand_scores):
            missing = set(cand_scores) - set(scores)
            if not missing:
                missing = set(scores) - set(cand_scores)
            raise GateError('PAIRING ' + comp + ' ' + str(min(missing)))
        comp_scores[comp] = scores

    regression_path = os.path.join(rundir, 'regression.jsonl')
    if not os.path.exists(regression_path):
        raise GateError('NO_REGRESSION')
    regression = load_jsonl(regression_path)

    return {
        'candidates': candidates,
        'primary': primary,
        'metrics': metrics,
        'cand_scores': cand_scores,
        'audit': audit,
        'comp_scores': comp_scores,
        'regression': regression,
        'rundir': rundir,
    }


def cmd_vet(rundir, run_tag='untagged') -> int:
    try:
        data = load_rundir(rundir)
    except GateError as e:
        print('VET_FAIL ' + str(e))
        return 1
    print('VET_OK')
    return 0


import math
import random


def paired_bootstrap_ci(deltas, n_resamples, seed, conf=0.95):
    n = len(deltas)
    if n == 0:
        raise GateError("paired bootstrap requires at least one delta")
    if not isinstance(n_resamples, int) or isinstance(n_resamples, bool) or n_resamples <= 0:
        raise GateError("n_resamples must be a positive integer")
    if not 0.0 < conf < 1.0:
        raise GateError("conf must be strictly between 0 and 1")

    rng = random.Random(seed)
    means = []
    for _ in range(n_resamples):
        picks = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(picks) / n)

    means.sort()
    lo_idx = max(0, math.floor((0.5 * (1.0 - conf)) * len(means)))
    hi_idx = min(
        len(means) - 1,
        math.ceil((1.0 - 0.5 * (1.0 - conf)) * len(means)) - 1,
    )
    return (means[lo_idx], means[hi_idx])


def comparator_deltas(cand_scores, comp_scores, ids):
    return [cand_scores[i] - comp_scores[i] for i in ids]


def rule_of_three(n) -> float:
    return 1.0 - 0.05 ** (1.0 / max(n, 1))


import json
import os


def cmd_verdict(rundir, run_tag='untagged') -> int:
    try:
        data = load_rundir(rundir)
    except GateError as e:
        print('VET_FAIL ' + str(e))
        return 1

    ids = sorted(data['cand_scores'].keys())
    lower_bounds = {}
    fail = []
    for comp in GATE_CONFIG['comparators']:
        deltas = comparator_deltas(
            data['cand_scores'], data['comp_scores'][comp], ids)
        lo, hi = paired_bootstrap_ci(
            deltas, GATE_CONFIG['bootstrap_rounds'],
            GATE_CONFIG['seed'], GATE_CONFIG['ci'])
        lower_bounds[comp] = lo
        if lo <= 0.0:
            fail.append('BASE_COMPARATOR ' + comp + ' lower=' + str(lo))

    for row in data['regression']:
        if not (row.get('expected_pass') is True and
                row.get('candidate_pass') is True):
            fail.append('REGRESSION ' + str(row.get('case_id')))

    critical_rows = [r for r in data['audit'] if r.get('critical') is True]
    if len(critical_rows) > GATE_CONFIG['critical_fabrications_allowed']:
        fail.append('CRITICAL_FABRICATION ' + str(len(critical_rows)))

    fabricated_classes = (
        'nonexistent-citation', 'nonexistent-commit', 'wrong-repository',
        'misattributed', 'real-but-non-supporting',
        'unsupported-factual-asserted-as-sourced')
    for row in data['audit']:
        finding = str(row.get('finding', ''))
        if (any(cls in finding for cls in fabricated_classes) and
                row.get('critical') is not True):
            fail.append('EXISTS_NOT_SUPPORT ' + str(row.get('item_id')))

    file_paths = [
        os.path.join(rundir, 'candidates.json'),
        os.path.join(rundir, data['primary']['metrics_file']),
        os.path.join(rundir, data['primary']['peritem_file']),
        os.path.join(rundir, data['primary']['audit_file']),
    ]
    for comp in GATE_CONFIG['comparators']:
        file_paths.append(os.path.join(
            rundir, 'baseline', comp, 'peritem.jsonl'))
    file_paths.append(os.path.join(rundir, 'regression.jsonl'))

    inputs_sha256 = {}
    for path in file_paths:
        with open(path, 'rb') as handle:
            inputs_sha256[path] = sha256_hex(handle.read())

    n_audited = len(data['audit'])
    bound = rule_of_three(n_audited)
    audit_note = None
    if n_audited < GATE_CONFIG['audit_min_blind_sample']:
        audit_note = ('AUDIT_BELOW_MIN_SAMPLE observed n=%d < min %d; '
                      'rule-of-three bound not meaningful') % (
            n_audited, GATE_CONFIG['audit_min_blind_sample'])

    receipt = {
        'gate': 'frontier_promotion_v1',
        'run_tag': run_tag,
        'verdict': 'PASS' if not fail else 'FAIL',
        'fail_reasons': fail,
        'critical_fabrications': len(critical_rows),
        'n_audited': n_audited,
        'bootstrap': {
            'rounds': GATE_CONFIG['bootstrap_rounds'],
            'ci': GATE_CONFIG['ci'],
            'seed': GATE_CONFIG['seed'],
            'lower_bounds': lower_bounds,
        },
        'inputs_sha256': inputs_sha256,
        'audit_note': audit_note,
        'rule_of_three_upper': bound,
        'rule_of_three_text': (
            '0 observed in %d audited, upper bound ~%.2f%%' %
            (n_audited, bound * 100.0)),
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not fail else 1


import json
import os


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def build_synth_rundir(base_dir, name, *, unpaired=False, metric_mismatch=False,
                       miss_regression=False, critical_fabrication=False,
                       beat_baseline_only=False, audit_unknown=False):
    rd = os.path.join(base_dir, name)
    os.makedirs(rd, exist_ok=True)

    candidates = [{
        "model_id": "synth-4b",
        "metrics_file": "metrics.json",
        "peritem_file": "peritem.jsonl",
        "audit_file": "audit.jsonl",
    }]
    with open(os.path.join(rd, "candidates.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(candidates, handle)

    metrics = {"primary_metric": "wrong_metric" if metric_mismatch else "heldout_exact"}
    with open(os.path.join(rd, "metrics.json"), "w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle)

    candidate_rows = [
        {"item_id": i, "score": 0.70 + (i % 7) * 0.02}
        for i in range(40)
    ]
    if unpaired:
        candidate_rows.append({"item_id": 999, "score": 0.9})
    _write_jsonl(os.path.join(rd, "peritem.jsonl"), candidate_rows)

    frozen_dir = os.path.join(rd, "baseline", "frozen_0_8b")
    os.makedirs(frozen_dir, exist_ok=True)
    frozen_rows = [
        {"item_id": i, "score": 0.55 + (i % 5) * 0.03}
        for i in range(40)
    ]
    _write_jsonl(os.path.join(frozen_dir, "peritem.jsonl"), frozen_rows)

    unadapted_dir = os.path.join(rd, "baseline", "unadapted_base")
    os.makedirs(unadapted_dir, exist_ok=True)
    unadapted_start = 0.72 if beat_baseline_only else 0.60
    unadapted_rows = [
        {"item_id": i, "score": unadapted_start + (i % 5) * 0.03}
        for i in range(40)
    ]
    _write_jsonl(os.path.join(unadapted_dir, "peritem.jsonl"), unadapted_rows)

    regression_rows = []
    for i in range(10):
        regression_rows.append({
            "case_id": "case%d" % i,
            "expected_pass": True,
            "candidate_pass": not (miss_regression and i == 7),
        })
    _write_jsonl(os.path.join(rd, "regression.jsonl"), regression_rows)

    audit_rows = [
        {"item_id": item_id, "checker": "deterministic", "critical": False, "finding": "clean"}
        for item_id in [1, 2, 3, 4, 5]
    ]
    if critical_fabrication:
        audit_rows.append({
            "item_id": 6,
            "checker": "deterministic",
            "critical": True,
            "finding": "nonexistent-citation",
        })
    if audit_unknown:
        audit_rows.append({
            "item_id": 5555,
            "checker": "deterministic",
            "critical": False,
            "finding": "clean",
        })
    _write_jsonl(os.path.join(rd, "audit.jsonl"), audit_rows)

    return rd


import contextlib
import io


def run_verdict_capture(rundir):
    with contextlib.redirect_stdout(io.StringIO()) as buf:
        rc = cmd_verdict(rundir)
        return rc, buf.getvalue()


def c1_clean_pass(tmp):
    rd = build_synth_rundir(tmp, 'c1')
    rc, out = run_verdict_capture(rd)
    ok = rc == 0 and 'PASS' in out and 'FAIL' not in out
    return ('c1_clean_pass', ok, 'rc=' + str(rc))


def c2_unpaired(tmp):
    rd = build_synth_rundir(tmp, 'c2', unpaired=True)
    rc, out = run_verdict_capture(rd)
    ok = 'PAIRING' in out and rc != 0
    return ('c2_unpaired', ok, 'rc=' + str(rc))


def c3_metric_mismatch(tmp):
    rd = build_synth_rundir(tmp, 'c3', metric_mismatch=True)
    rc, out = run_verdict_capture(rd)
    ok = 'METRIC_MISMATCH' in out and rc != 0
    return ('c3_metric_mismatch', ok, 'rc=' + str(rc))


def c4_missed_regression(tmp):
    rd = build_synth_rundir(tmp, 'c4', miss_regression=True)
    rc, out = run_verdict_capture(rd)
    ok = 'REGRESSION' in out and rc != 0
    return ('c4_missed_regression', ok, 'rc=' + str(rc))


def c5_critical_fabrication(tmp):
    rd = build_synth_rundir(tmp, 'c5', critical_fabrication=True)
    rc, out = run_verdict_capture(rd)
    ok = 'CRITICAL_FABRICATION' in out and rc != 0
    return ('c5_critical_fabrication', ok, 'rc=' + str(rc))


import json
import shutil
import tempfile


def c6_base_comparator(tmp):
    rd = build_synth_rundir(tmp, 'c6', beat_baseline_only=True)
    rc, out = run_verdict_capture(rd)
    ok = ('BASE_COMPARATOR' in out and rc != 0)
    return ('c6_base_comparator', ok, 'rc=' + str(rc))


def c7_bootstrap_determinism(_):
    deltas = [0.1, -0.2, 0.3, 0.05, -0.1, 0.2, 0.25, 0.3, 0.15, -0.05]
    a = paired_bootstrap_ci(deltas, 500, 20261120, 0.95)
    b = paired_bootstrap_ci(deltas, 500, 20261120, 0.95)
    ok = (a == b)
    return ('c7_bootstrap_determinism', ok, str(a) + ' / ' + str(b))


def c8_receipt_hash_changes(tmp):
    rd1 = build_synth_rundir(tmp, 'c8a')
    rd2 = build_synth_rundir(tmp, 'c8b')
    peritem_path = os.path.join(rd2, 'peritem.jsonl')
    rows = load_jsonl(peritem_path)
    rows[0]['score'] = rows[0]['score'] + 0.01
    _write_jsonl(peritem_path, rows)
    rc1, out1 = run_verdict_capture(rd1)
    rc2, out2 = run_verdict_capture(rd2)
    r1 = json.loads(out1)
    r2 = json.loads(out2)
    ok = (r1['inputs_sha256'] != r2['inputs_sha256'])
    return ('c8_receipt_hash_changes', ok, 'sha sets differ=' + str(ok))


def c9_rule_of_three(_):
    v = rule_of_three(200)
    ok = (abs(v - (1 - 0.05 ** (1.0 / 200))) < 1e-12 and 1.46 <= v * 100.0 <= 1.53)
    return ('c9_rule_of_three', ok, '%.4f%%' % (v * 100.0))


def c10_audit_unknown(tmp):
    rd = build_synth_rundir(tmp, 'c10', audit_unknown=True)
    rc, out = run_verdict_capture(rd)
    ok = ('AUDIT_UNKNOWN_ITEM' in out and rc != 0)
    return ('c10_audit_unknown', ok, 'rc=' + str(rc))


def cmd_selftest(run_tag='untagged') -> int:
    tmp = tempfile.mkdtemp(prefix='gate_selftest_')
    results = []
    for fn in (
        c1_clean_pass,
        c2_unpaired,
        c3_metric_mismatch,
        c4_missed_regression,
        c5_critical_fabrication,
        c6_base_comparator,
        c7_bootstrap_determinism,
        c8_receipt_hash_changes,
        c9_rule_of_three,
        c10_audit_unknown,
    ):
        results.append(fn(tmp))
    for name, ok, detail in results:
        print('CTRL %s %s %s' % (name, 'OK' if ok else 'FAIL', detail))
    n_ok = sum(1 for r in results if r[1])
    print('SELFTEST %d/%d OK' % (n_ok, 10))
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if n_ok == 10 else 1


if __name__ == '__main__':
    sys.exit(main())
