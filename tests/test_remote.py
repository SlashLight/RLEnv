"""Offline tests only: never contact the challenge or start a model."""
import json
import subprocess
import threading
from unittest.mock import Mock

import pytest

from pipeline import docker, remote
from pipeline.config import Config
from pipeline.evaluate import evaluate
from pipeline.runner import Runner, RunnerError
from pipeline.util import tree_manifest, write_json


@pytest.mark.parametrize('reference,submitted,verdict', [
    ('', 'anything', 'not_evaluated'),
    (' secret ', '\nsecret\t', 'confirmed'),
    ('secret', 'wrong', 'not_confirmed'),
    ('secret', None, 'not_confirmed'),
])
def test_remote_evaluation(tmp_path, reference, submitted, verdict):
    write_json(tmp_path / 'passport.json', {'remote_url': remote.URL})
    write_json(tmp_path / 'status.json', {'status': 'done'})
    write_json(tmp_path / 'submission_manifest.json', {'health': 'ok', 'present': True})
    (tmp_path / 'private').mkdir()
    (tmp_path / 'private/expected_flag.txt').write_text(reference + '\n')
    (tmp_path / 'submission').mkdir()
    if submitted is not None:
        (tmp_path / 'submission/flag.txt').write_text(submitted)
    write_json(tmp_path / 'submission_manifest.json', {
        'health': 'ok', 'present': True, 'files': tree_manifest(tmp_path / 'submission'),
    })
    factory = Mock(side_effect=AssertionError('replay forbidden'))
    report = evaluate(tmp_path, replay_env_factory=factory, new_flag='new')
    factory.assert_not_called()
    assert report['flag_check']['verdict'] == verdict
    assert report['solution']['status'] == verdict
    assert report['script_check']['verdict'] == 'not_evaluated'
    assert 'expected_flag' not in json.dumps(report)


def test_reject_other_targets_before_start(tmp_path):
    runner = Runner(Config('p', 'm', runs_dir=tmp_path))
    with pytest.raises(RunnerError):
        runner.start({'remote': {'url': 'http://other:21023'}})
    assert not list(tmp_path.iterdir())


def test_relay_fixed_destination(monkeypatch):
    calls = []
    monkeypatch.setattr(remote.socket, 'gethostbyname', lambda host: '192.0.2.1')
    def fake(*args, **kwargs):
        calls.append(args)
        output = b'true\n' if args[0] == 'inspect' else b'READY\n'
        return subprocess.CompletedProcess(args, 0, output, b'')
    monkeypatch.setattr(docker, 'docker', fake)
    remote.start_relay('tools', 'isolated', 'relay')
    assert calls[0][-1] == '192.0.2.1'
    assert '--cap-drop' in calls[0]
    assert ('network', 'connect', '--alias', remote.HOST, 'isolated', 'relay') in calls
    assert 'socket.create_connection(upstream' in remote.RELAY
    assert 'sys.argv[1], 21023' in remote.RELAY


@pytest.mark.parametrize('relay_fails', [False, True])
def test_remote_lifecycle(tmp_path, monkeypatch, relay_fails):
    import pipeline.runner as module
    cfg = Config('p', 'm', runs_dir=tmp_path)
    runner = Runner(cfg)
    root = tmp_path / 'r1'
    write_json(root / 'status.json', {'run_id': 'r1', 'status': 'preparing'})
    runner._stop_events['r1'] = threading.Event()
    sandbox = Mock(tools_image_id='tools-id', tools_image='tools', network='isolated')
    sandbox.cleanup.return_value = []
    sandbox.snapshot_submission.return_value = {
        'present': False, 'files': [], 'violations': [], 'health': 'ok',
    }
    session = Mock(saw_tool_call=True)
    session.prompt.return_value = {'success': True, 'data': {'disposition': 'queued'}}
    session.wait_settled.return_value = 'settled'
    monkeypatch.setattr(module, 'Sandbox', lambda *a: sandbox)
    monkeypatch.setattr(module, 'PiSession', lambda *a: session)
    monkeypatch.setattr(module, 'Gateway', lambda *a: Mock(url='gateway', token='token'))
    monkeypatch.setattr(module, '_tool_version', lambda *a: 'test')
    for name in ['load_task', 'prepare_build_context']:
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError(name)))
    for name in ['build_task_image', 'start_service', 'check_ready', 'remove_image']:
        monkeypatch.setattr(docker, name, Mock(side_effect=AssertionError(name)))
    remove = Mock()
    monkeypatch.setattr(docker, 'remove_container', remove)
    relay = Mock(side_effect=RuntimeError('relay failed') if relay_fails else None)
    monkeypatch.setattr(remote, 'start_relay', relay)
    runner._execute('r1', cfg, {'url': remote.URL, 'expected_flag': 'PRIVATE_REFERENCE'})
    status = json.loads((root / 'status.json').read_text())
    assert status['status'] == ('failed' if relay_fails else 'done')
    remove.assert_called_with('pipeline-r1-task')
    sandbox.cleanup.assert_called_once()
    assert (root / 'private/expected_flag.txt').read_text().strip() == 'PRIVATE_REFERENCE'
    assert 'PRIVATE_REFERENCE' not in (root / 'passport.json').read_text()
    assert 'PRIVATE_REFERENCE' not in (root / 'instructions.txt').read_text()
    assert not json.loads((root / 'passport.json').read_text())['replay_enabled']
    if not relay_fails:
        runner.reassess('r1')
        assert not (root / 'workspace').exists()
        session.prompt.assert_called_once()
    else:
        session.start.assert_not_called()
