"""Regression checks for the runner/container/package boundaries."""
import json
import subprocess
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline import docker
from pipeline.config import Config
from pipeline.evaluate import evaluate
from pipeline.replay import DockerReplayEnvironment
from pipeline.runner import Runner
from pipeline.task import load_task
from pipeline.util import ProcessResult, tree_manifest, write_json


def package(root):
    write_json(root / 'passport.json', {'run_id': 'r1'})
    write_json(root / 'status.json', {'status': 'done'})
    write_json(root / 'submission_manifest.json', {
        'present': True, 'health': 'ok', 'violations': [], 'files': [],
    })
    (root / 'submission').mkdir()
    (root / 'submission' / 'solution.py').write_text('print(1)')
    write_json(root / 'submission_manifest.json', {
        'present': True, 'health': 'ok', 'violations': [],
        'files': tree_manifest(root / 'submission'),
    })
    return root


@pytest.mark.parametrize('contents', [None, '{broken json\n'])
def test_missing_or_corrupt_journal_is_not_complete(tmp_path, contents):
    root = package(tmp_path)
    if contents is not None:
        (root / 'events.jsonl').write_text(contents)
    report = evaluate(root, execute_replay=False)
    assert report['data_health']['complete'] is False
    assert report['data_health']['recording_errors']


def test_docker_launch_failure_is_check_error(tmp_path, monkeypatch):
    root = package(tmp_path)
    env = DockerReplayEnvironment(Config('p', 'm'),
                                  load_task('my-flask-app', Path('tasks')), root)
    env.tools_image = 'tools'
    monkeypatch.setattr(env, 'prepare', lambda flag: 'http://task')
    monkeypatch.setattr(env, 'close', lambda: [])
    monkeypatch.setattr('pipeline.replay.run_bounded', lambda *a, **kw:
                        ProcessResult(125, b'', b'daemon unavailable'))
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess([], 125, b'', b'daemon unavailable'))
    report = evaluate(root, replay_env_factory=lambda: env, new_flag='fresh')
    assert report['script_check']['verdict'] == 'check_error'


@pytest.mark.parametrize('remove', [docker.remove_container, docker.remove_network, docker.remove_image])
def test_cleanup_does_not_swallow_daemon_failure(monkeypatch, remove):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess([], 1, b'', b'daemon unavailable'))
    with pytest.raises(docker.DockerError):
        remove('test-resource')


@pytest.mark.parametrize('evaluation_fails', [True, False])
def test_runner_stops_before_snapshot_and_preserves_materials(tmp_path, monkeypatch, evaluation_fails):
    import pipeline.runner as module
    cfg = Config('p', 'm', runs_dir=tmp_path)
    runner = Runner(cfg)
    root = tmp_path / 'r1'
    write_json(root / 'status.json', {'run_id': 'r1', 'status': 'preparing'})
    runner._stop_events['r1'] = threading.Event()
    sandbox = Mock()
    sandbox.tools_image_id = 'tools-id'
    sandbox.tools_image = 'tools'
    sandbox.network = 'network'
    sandbox.cleanup.return_value = []
    sandbox.snapshot_submission.return_value = {
        'present': False, 'files': [], 'violations': [], 'health': 'ok',
    }
    session = Mock()
    session.prompt.return_value = {'success': True, 'data': {'disposition': 'queued'}}
    session.wait_settled.return_value = 'settled'
    session.saw_tool_call = True
    gateway = Mock(url='http://gateway', token='token')
    monkeypatch.setattr(module, 'Sandbox', lambda *a: sandbox)
    monkeypatch.setattr(module, 'PiSession', lambda *a: session)
    monkeypatch.setattr(module, 'Gateway', lambda *a: gateway)
    monkeypatch.setattr(module, '_tool_version', lambda *a: 'test')
    monkeypatch.setattr(docker, 'build_task_image', lambda *a, **kw: 'task-id')
    monkeypatch.setattr(docker, 'start_service', lambda *a, **kw: None)
    monkeypatch.setattr(docker, 'check_ready', lambda *a, **kw: None)
    monkeypatch.setattr(docker, 'remove_container', lambda *a, **kw: None)
    monkeypatch.setattr(docker, 'remove_image', lambda *a, **kw: None)
    evaluator = Mock(side_effect=RuntimeError('evaluator broken') if evaluation_fails else None,
                     return_value={})
    monkeypatch.setattr(module, 'evaluate', evaluator)
    runner._execute('r1', cfg)
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == ('failed' if evaluation_fails else 'done')
    if evaluation_fails:
        assert 'evaluator broken' in status['evaluation_error']
    assert evaluator.call_count == 1
    assert (root / 'author_materials' / 'app' / 'app.py').read_bytes() == Path(
        'tasks/my-flask-app/dist/app/app.py').read_bytes()
    assert not (root / 'workspace').exists()
    assert sandbox.method_calls.index(next(c for c in sandbox.method_calls if c[0] == 'stop')) < \
        sandbox.method_calls.index(next(c for c in sandbox.method_calls if c[0] == 'snapshot_submission'))
