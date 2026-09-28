from itertools import islice
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from benchmark.preencode_hevc import collect_videos, encode_videos, main, task_names


class Task:
    def __init__(self, paths):
        self.eval_docs = paths

    def doc_iterator(self, *, limit):
        return islice(enumerate(self.eval_docs), limit)

    def doc_to_messages(self, path):
        return [{'role': 'user', 'content': [{'type': 'video', 'url': str(path)}]}]


def test_task_union_and_per_leaf_limit_deduplicate_videos(tmp_path):
    paths = [tmp_path / str(i) for i in range(4)]
    for path in paths:
        path.touch()
    tasks = [SimpleNamespace(task_name='a', task=Task(paths)),
             SimpleNamespace(task_name='b', task=Task([paths[1], paths[2], paths[3]]))]
    videos, counts = collect_videos(tasks, 2)
    assert list(videos) == paths[:3]
    assert counts == {'a': 2, 'b': 2}
    assert len(videos[paths[1]]) == 2
    videos, counts = collect_videos(tasks, 0.25)
    assert list(videos) == paths[:2]
    assert counts == {'a': 1, 'b': 1}
    assert task_names(['a,b', 'b c']) == ['a', 'b', 'c']


def test_missing_source_stops_before_any_encoding(tmp_path):
    task = SimpleNamespace(task_name='missing', task=Task([tmp_path / 'absent.mp4']))
    with pytest.raises(FileNotFoundError, match='missing/0'):
        collect_videos([task], 1)


def test_encoding_counts_hits_and_continues_after_failure(tmp_path):
    artifact = dict(cache_key='key', path=tmp_path / 'encoded.mp4',
                    actual_encoder='libx265', paid_encode_ms=0)
    store = Mock()
    store.get_or_encode.side_effect = [SimpleNamespace(cache_hit=True, **artifact),
                                      RuntimeError('bad source'),
                                      SimpleNamespace(cache_hit=False, **artifact)]
    paths = [tmp_path / str(i) for i in range(3)]
    rows = encode_videos(paths, store, 32)
    assert [row['status'] for row in rows] == ['hit', 'failed', 'encoded']
    assert 'bad source' in rows[1]['error']
    store.get_or_encode.assert_any_call(paths[0], encode_scope='full-video', sampled_gop_size=32)


@pytest.mark.parametrize('scope,ablation', [('sampled-clip', 'none'), ('full-video', 'black')])
def test_rejects_model_dependent_preencoding(monkeypatch, scope, ablation):
    monkeypatch.setenv('HEVC_ENCODE_SCOPE', scope)
    monkeypatch.setenv('VIDEO_ABLATION', ablation)
    with pytest.raises(SystemExit) as exc:
        main(['--tasks', 'nextqa_mc_test'])
    assert exc.value.code == 2


@pytest.mark.parametrize('fail_preencode', [False, True])
def test_submission_queues_one_preencode_and_dependent_arrays(tmp_path, fail_preencode):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys

    fake = tmp_path / 'sbatch'
    log = tmp_path / 'submissions.jsonl'
    fake.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
with open(os.environ['SUBMISSION_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'tasks': os.getenv('TASKS'),
                             'task_list': os.getenv('TASK_LIST_IN'),
                             'store': os.getenv('HEVC_PERMANENT_DIR')}) + '\\n')
if '--parsable' in sys.argv:
    if os.getenv('FAIL_PREENCODE') == '1': sys.exit(1)
    print('12345;cluster')
else:
    print('Submitted batch job 23456')
''')
    fake.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / 'scripts/submit_vlm_last.sh'
    environment = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
                       PROJECT_ROOT=str(tmp_path), SUBMISSION_LOG=str(log),
                       FAIL_PREENCODE=str(int(fail_preencode)))
    for key in ('QWEN_TASKS', 'HF_TASKS', 'OFF_TASKS', 'LNV_TASKS', 'CONCURRENT'):
        environment.pop(key, None)
    result = subprocess.run(['bash', str(script)], env=environment, capture_output=True, text=True)
    submissions = [json.loads(line) for line in log.read_text().splitlines()]
    if fail_preencode:
        assert result.returncode != 0
        assert len(submissions) == 1
    else:
        assert result.returncode == 0, result.stderr
        assert len(submissions) == 5
        assert set(task_names([submissions[0]['tasks']])) == {
            'nextqa', 'motionbench', 'mvbench', 'vitatecs'}
        for row, count in zip(submissions[1:], (4, 4, 4, 3)):
            assert '--dependency=afterok:12345' in row['args']
            assert f'--array=1-{count}%4' in row['args']
            assert row['store'] == submissions[0]['store']
            wrap = next(arg.removeprefix('--wrap=') for arg in row['args'] if arg.startswith('--wrap='))
            subprocess.run(['sh', '-n'], input=wrap, text=True, check=True)
