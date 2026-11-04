"""Container operations use controlled subprocess outcomes, never a live daemon."""
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline import docker
from pipeline.config import Config
from pipeline.evaluate import check_flag, check_script
from pipeline.replay import DockerReplayEnvironment
from pipeline.runlog import EventLog
from pipeline.sandbox import Sandbox, SandboxError
from pipeline.task import load_task
from pipeline.util import ProcessResult


def test_preparation_budget_caps_every_docker_call_and_expires(monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(docker.time, 'monotonic', lambda: clock[0])
    run = Mock(return_value=subprocess.CompletedProcess([], 0, b'image-id', b''))
    monkeypatch.setattr(subprocess, 'run', run)
    with docker.deadline(15):
        docker.image_id('tools')  # formerly had no timeout
        assert run.call_args.kwargs['timeout'] == 5
        clock[0] = 13
        docker.docker('build', timeout=900)
        assert run.call_args.kwargs['timeout'] == 2
        clock[0] = 15
        with pytest.raises(docker.DockerError, match='deadline'):
            docker.docker('run')
    docker.docker('version', timeout=30)
    assert run.call_args.kwargs['timeout'] == 30


@pytest.mark.parametrize('remove, message', [
    (docker.remove_container, b'Error response from daemon: No such container: test'),
    (docker.remove_network, b'Error response from daemon: network test not found'),
    (docker.remove_image, b'Error response from daemon: No such image: test'),
])
def test_cleanup_is_idempotent_for_absent_resources(monkeypatch, remove, message):
    monkeypatch.setattr(subprocess, 'run', lambda *a, **kw:
                        subprocess.CompletedProcess([], 1, b'', message))
    remove('test')


@pytest.mark.parametrize('code', [0, 125, 126, 127])
def test_replay_uses_container_exit_not_docker_cli_code(tmp_path, monkeypatch, code):
    submission = tmp_path / 'submission'
    submission.mkdir()
    (submission / 'solution.py').write_text('pass')
    env = DockerReplayEnvironment(Config('p', 'm'),
                                  load_task('my-flask-app', Path('tasks')), tmp_path)
    env.tools_image = 'tools'
    def run(command, **kwargs):
        if command[1] == 'inspect':
            state = {'Status': 'exited', 'ExitCode': code, 'Error': '', 'OOMKilled': False}
            return subprocess.CompletedProcess(command, 0, json.dumps(state).encode(), b'')
        return ProcessResult(code, b'output', b'diagnostics')
    monkeypatch.setattr(subprocess, 'run', run)
    monkeypatch.setattr('pipeline.replay.run_bounded', run)
    result = env.run(submission, 'http://task', 1)
    assert result.exit_code == code
    assert result.stdout == 'output'
    assert not result.timed_out


@pytest.mark.parametrize('root_kind', ['directory', 'symlink', 'missing', 'daemon_error'])
def test_submission_snapshot_from_stopped_container(tmp_path, monkeypatch, root_kind):
    log = EventLog(tmp_path / 'run', 'r1')
    sandbox = Sandbox(Config('p', 'm'), 'r1', tmp_path / 'workspace', log, Path('pipeline'))
    task = load_task('my-flask-app', Path('tasks'))
    dest = tmp_path / 'run' / 'submission'
    try:
        with pytest.raises(SandboxError, match='stopped'):
            sandbox.snapshot_submission(task, dest)
        stopped = []
        monkeypatch.setattr(docker, 'stop_container', lambda name: stopped.append(name))
        sandbox.stop()
        def copy(container, src, target, **kwargs):
            assert stopped == [container]
            assert src == '/workspace/submission'  # root symlinks must not be dereferenced
            if root_kind == 'missing':
                raise docker.DockerError(f'Could not find the file {src} in container {container}')
            if root_kind == 'daemon_error':
                raise docker.DockerError('daemon unavailable')
            copied = target / 'submission'
            if root_kind == 'symlink':
                copied.symlink_to('/tmp', target_is_directory=True)
            else:
                copied.mkdir()
                (copied / 'flag.txt').write_text('flag')
                (copied / 'link').symlink_to('/etc/passwd')
                (copied / 'dir-link').symlink_to(tmp_path, target_is_directory=True)
                (copied / 'nested').mkdir()
                (copied / 'nested' / 'helper.txt').write_text('helper')
                os.mkfifo(copied / 'nested' / 'pipe')
        monkeypatch.setattr(docker, 'copy_from_container', copy)
        manifest = sandbox.snapshot_submission(task, dest)
        if root_kind == 'directory':
            assert (dest / 'flag.txt').read_text() == 'flag'
            assert not (dest / 'link').is_symlink()
            assert not (dest / 'dir-link').is_symlink()
            assert not (dest / 'nested' / 'pipe').exists()
            assert (dest / 'nested' / 'helper.txt').read_text() == 'helper'
            assert {v['path'] for v in manifest['violations']} == {
                'link', 'dir-link', 'nested/pipe',
            }
            assert all(v['kind'] == 'not_a_regular_file' for v in manifest['violations'])
            assert all(f['type'] == 'file' for f in manifest['files'])
            package = {'run_dir': dest.parent, 'expected_flag': 'flag',
                       'submission_manifest': manifest}
            assert check_flag(package)['verdict'] == 'confirmed'
            replay = check_script(package, None, 1, 'fresh', dest.parent / 'checks')
            assert replay['verdict'] == 'not_confirmed'
            assert replay['reason'] == 'invalid_submission'
        elif root_kind == 'symlink':
            assert manifest['violations'][0]['type'] == 'invalid_submission_root'
            assert not dest.exists()
        elif root_kind == 'missing':
            assert manifest['present'] is False and manifest['health'] == 'ok'
        else:
            assert manifest['health'] == 'error'
    finally:
        log.close()


def test_readiness_timeout_cleans_container_outside_expired_budget(monkeypatch):
    calls = []
    clock = [0.0]
    monkeypatch.setattr(docker.time, 'monotonic', lambda: clock[0])
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == 'run':
            clock[0] = 10
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        return subprocess.CompletedProcess(command, 0, b'', b'')
    monkeypatch.setattr(subprocess, 'run', run)
    with docker.deadline(5), pytest.raises(docker.DockerError, match='timed out'):
        docker.check_ready('network', 'http://task', 200, 'tools', 30)
    assert calls[0][1]['timeout'] == 5
    name = calls[0][0][calls[0][0].index('--name') + 1]
    assert calls[1][0] == ['docker', 'rm', '-f', name]
    assert calls[1][1]['timeout'] == 30


def test_notebook_observer_displays_result_without_duplicate_sandbox_output():
    notebook = json.loads(Path('notebooks/evaluator-pipeline.ipynb').read_text())
    assert any('Observer(api, RUN_ID)' in ''.join(cell['source']) for cell in notebook['cells'])
    from pipeline.notebook import Observer
    events = [
        {'seq': 1, 'source': 'pi', 'type': 'tool_execution_end',
         'data': {'result': {'content': [{'type': 'text', 'text': 'HTTP result 200'}]},
                  'isError': False}},
        {'seq': 2, 'source': 'sandbox', 'type': 'tool_exec',
         'data': {'output': 'sandbox output', 'exit_code': 0}},
    ]
    observer = Observer(None, 'r1')
    rendered = ''.join(observer.update(events))
    assert 'HTTP result 200' in rendered
    assert 'sandbox output' not in rendered
    assert observer.after == 2
