"""Audit regressions: offline only; no Docker daemon or model calls."""
import json
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline.config import Config
from pipeline.evaluate import build_data_health, evaluate
from pipeline.runner import Runner, RunnerError
from pipeline.util import tree_manifest, write_json


def package(root):
    write_json(root / 'passport.json', {'remote_url': 'http://task.duckerz.ru:21023'})
    write_json(root / 'status.json', {'status': 'done'})
    (root / 'private').mkdir()
    (root / 'private/expected_flag.txt').write_text('secret\n')
    (root / 'submission').mkdir()
    (root / 'submission/flag.txt').write_text('secret')
    write_json(root / 'submission_manifest.json', {
        'health': 'ok', 'present': True, 'violations': [],
        'files': tree_manifest(root / 'submission'),
    })
    return root


@pytest.mark.parametrize('damage', ['missing', 'changed', 'added'])
@pytest.mark.parametrize('local', [False, True])
def test_damaged_submission_is_not_graded(tmp_path, damage, local):
    root = package(tmp_path / 'r1')
    if local:
        write_json(root / 'passport.json', {'schema': 'evaluator-pipeline/passport/v2'})
    flag = root / 'submission/flag.txt'
    if damage == 'missing':
        flag.unlink()
    elif damage == 'changed':
        flag.write_text('wrong!')
    else:
        (root / 'submission/solution.py').write_text('pass')
    env = Mock()
    report = evaluate(root, replay_env_factory=lambda: env, new_flag='fresh')
    assert report['flag_check']['verdict'] == 'check_error'
    assert report['flag_check']['reason'] == 'submission_integrity_error'
    if local:
        assert report['script_check']['reason'] == 'submission_integrity_error'
    env.prepare.assert_not_called()


def test_raw_references_must_exist_and_match(tmp_path):
    (tmp_path / 'raw').mkdir()
    event = {'seq': 1, 'type': 'message_end', 'raw_line': 1,
             'data': {'type': 'message_end', 'message': {'role': 'assistant', 'content': 'hi'}}}
    (tmp_path / 'events.jsonl').write_text(json.dumps(event) + '\n')
    raw = tmp_path / 'raw/rpc.jsonl'
    for content in ['', '{"type":"agent_start"}\n']:
        raw.write_text(content)
        assert not build_data_health(tmp_path, [event])['complete']
    raw.write_text(json.dumps(event['data']) + '\n')
    assert build_data_health(tmp_path, [event])['complete']


def test_repeat_evaluation_excludes_other_work(tmp_path, monkeypatch):
    package(tmp_path / 'r1')
    runner = Runner(Config('p', 'm', runs_dir=tmp_path))
    entered, release = threading.Event(), threading.Event()
    def evaluating(*args):
        entered.set()
        assert release.wait(3)
        return {}
    monkeypatch.setattr(runner, '_evaluate_package', evaluating)
    worker = threading.Thread(target=runner.reassess, args=('r1',))
    worker.start()
    try:
        assert entered.wait(3)
        with pytest.raises(RunnerError):
            runner.reassess('r1')
        with pytest.raises(RunnerError):
            runner.start()
    finally:
        release.set()
        worker.join(3)


def test_unfinished_package_cannot_be_reassessed(tmp_path, monkeypatch):
    root = package(tmp_path / 'r1')
    write_json(root / 'status.json', {'status': 'running'})
    runner = Runner(Config('p', 'm', runs_dir=tmp_path))
    evaluator = Mock()
    monkeypatch.setattr(runner, '_evaluate_package', evaluator)
    with pytest.raises(RunnerError):
        runner.reassess('r1')
    evaluator.assert_not_called()


def test_process_capture_is_bounded_and_marks_truncation():
    from pipeline.util import run_bounded
    result = run_bounded([sys.executable, '-c',
        'import os; os.write(1,b"x"*200000); os.write(2,b"y"*200000)'],
        timeout=3, output_limit=1024)
    assert result.returncode == 0
    assert result.stdout == b'x' * 1024 and result.stderr == b'y' * 1024
    assert result.stdout_truncated and result.stderr_truncated


def test_process_capture_timeout_keeps_bounded_output():
    from pipeline.util import run_bounded
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        run_bounded([sys.executable, '-c', 'import os\nwhile True: os.write(1,b"x"*65536)'],
                    timeout=0.2, output_limit=512)
    assert len(caught.value.stdout) <= 512


def test_event_poll_reads_only_new_bytes_and_resets(tmp_path, monkeypatch):
    runner = Runner(Config('p', 'm', runs_dir=tmp_path))
    root = tmp_path / 'r1'
    write_json(root / 'status.json', {'status': 'running'})
    journal = root / 'events.jsonl'
    def line(seq, text='x'):
        return json.dumps({'seq': seq, 'data': text}) + '\n'
    journal.write_text(''.join(line(n, 'x' * 1000) for n in range(1, 101)))
    assert len(runner.events('r1')) == 100
    # Counting JSON decodes catches a full reread without a timing threshold.
    loads = Mock(wraps=json.loads)
    monkeypatch.setattr('pipeline.runlog.json.loads', loads)
    assert runner.events('r1', 100) == []
    assert loads.call_count == 0
    with journal.open('a') as fh:
        fh.write(line(101))
    assert runner.events('r1', 100) == [{'seq': 101, 'data': 'x'}]
    assert loads.call_count == 1
    assert len(runner.events('r1', 0)) == 101
    journal.write_text(line(102, 'replacement'))
    assert runner.events('r1', 101)[0]['data'] == 'replacement'
    replacement = root / 'replacement'
    replacement.write_text(line(103))
    replacement.replace(journal)
    assert runner.events('r1', 102)[0]['seq'] == 103


def test_event_poll_retries_partial_line(tmp_path):
    runner = Runner(Config('p', 'm', runs_dir=tmp_path))
    root = tmp_path / 'r1'
    write_json(root / 'status.json', {'status': 'running'})
    journal = root / 'events.jsonl'
    journal.write_text('{"seq":1}\n{"seq":')
    assert runner.events('r1') == [{'seq': 1}]
    with journal.open('a') as fh:
        fh.write('2}\n')
    assert runner.events('r1', 1) == [{'seq': 2}]


@pytest.fixture
def lifecycle(tmp_path, monkeypatch):
    import pipeline.runner as module
    runner = Runner(Config('p', 'm', runs_dir=tmp_path))
    sandbox = Mock(tools_image='tools', tools_image_id='id', network='network')
    sandbox.cleanup.return_value = []
    sandbox.snapshot_submission.return_value = {
        'present': False, 'files': [], 'violations': [], 'health': 'ok',
    }
    session = Mock(saw_tool_call=False)
    session.prompt.return_value = {'success': True, 'data': {'disposition': 'queued'}}
    session.wait_settled.return_value = 'settled'
    relay = Mock()
    monkeypatch.setattr(module, 'Sandbox', lambda *a: sandbox)
    monkeypatch.setattr(module, 'PiSession', lambda *a, **kw: session)
    monkeypatch.setattr(module, 'Gateway', lambda *a: Mock(url='gateway', token='token'))
    monkeypatch.setattr(module, '_tool_version', lambda *a: 'test')
    monkeypatch.setattr(module.remote, 'start_relay', relay)
    monkeypatch.setattr(module.docker, 'remove_container', Mock())
    return runner, sandbox, session, relay


def start_remote(runner):
    return runner.start({'remote': {'url': 'http://task.duckerz.ru:21023'}})['run_id']


def test_stop_during_preparation_never_starts_model(lifecycle):
    runner, sandbox, session, relay = lifecycle
    entered, release = threading.Event(), threading.Event()
    def prepare(*args):
        entered.set()
        assert release.wait(3)
    relay.side_effect = prepare
    run_id = start_remote(runner)
    try:
        assert entered.wait(3)
        assert runner.stop(run_id)['stop'] == 'requested'
    finally:
        release.set()
        runner.shutdown()
    session.start.assert_not_called()
    session.prompt.assert_not_called()
    sandbox.cleanup.assert_called_once()
    assert runner.state(run_id)['stop_reason']['kind'] == 'user_stop'
    assert runner.report(run_id)['stop_reason']['kind'] == 'user_stop'


def test_stop_during_pi_start_never_sends_prompt(lifecycle):
    runner, sandbox, session, _ = lifecycle
    session.start.side_effect = lambda: next(iter(runner._stop_events.values())).set()
    run_id = start_remote(runner)
    # Wait for completion without shutdown requesting cancellation itself.
    with runner._idle:
        assert runner._idle.wait_for(lambda: not runner._busy, 3)
    session.start.assert_called_once()
    session.prompt.assert_not_called()
    session.close.assert_called_once()
    assert runner.state(run_id)['stop_reason']['kind'] == 'user_stop'


def test_shutdown_waits_for_cleanup_and_rejects_work(lifecycle):
    runner, sandbox, session, _ = lifecycle
    agent_running, cleaning, release = (threading.Event() for _ in range(3))
    def wait(deadline, stop):
        agent_running.set()
        assert stop.wait(3)
        return 'stopped'
    def cleanup():
        cleaning.set()
        assert release.wait(3)
        return []
    session.wait_settled.side_effect = wait
    sandbox.cleanup.side_effect = cleanup
    run_id = start_remote(runner)
    assert agent_running.wait(3)
    shutdown = threading.Thread(target=runner.shutdown)
    shutdown.start()
    try:
        assert cleaning.wait(3)
        assert shutdown.is_alive()
        with pytest.raises(RunnerError):
            runner.start()
    finally:
        release.set()
        shutdown.join(3)
    assert not shutdown.is_alive()
    assert runner.state(run_id)['status'] == 'done'
    assert runner.state(run_id)['stop_reason']['kind'] == 'user_stop'
    session.close.assert_called_once()
    assert not runner._threads


def test_server_interrupt_shuts_down_runner(monkeypatch):
    import pipeline.server as module
    runner, server = Mock(), Mock()
    server.serve_forever.side_effect = KeyboardInterrupt
    server.server_address = ('127.0.0.1', 1234)
    monkeypatch.setattr(module, 'Runner', lambda cfg: runner)
    monkeypatch.setattr(module, 'ThreadingHTTPServer', lambda *a: server)
    module.serve(Config('p', 'm'))
    runner.shutdown.assert_called_once()
    server.server_close.assert_called_once()


def test_replay_preparation_has_shared_deadline_and_separate_cleanup(tmp_path, monkeypatch):
    from pipeline import docker
    from pipeline.replay import DockerReplayEnvironment
    from pipeline.task import load_task
    cfg = Config('p', 'm')
    cfg.limits.prepare_seconds = 2
    pinned = 'sha256:' + 'a' * 64
    write_json(tmp_path / 'passport.json', {'environment': {'tools_image_id': pinned}})
    monkeypatch.setattr(docker, 'image_id', lambda image: pinned)
    env = DockerReplayEnvironment(cfg, load_task('my-flask-app', Path('tasks')), tmp_path)
    clock, calls = [10.0], []
    monkeypatch.setattr(docker.time, 'monotonic', lambda: clock[0])
    def run(command, **kwargs):
        calls.append((command, kwargs['timeout']))
        if command[1:3] == ['network', 'create']:
            clock[0] = 13.0
            raise subprocess.TimeoutExpired(command, kwargs['timeout'])
        clock[0] += 0.1
        return subprocess.CompletedProcess(command, 0, b'image-id', b'')
    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(docker.DockerError, match='timed out'):
        env.prepare('fresh')
    assert all(0 < timeout <= 2 for _, timeout in calls)
    count = len(calls)
    assert env.close() == []
    assert all(timeout == 30 for _, timeout in calls[count:])
    assert not env.build_dir.exists()


def test_sandbox_bounds_output_and_refuses_partial_file(tmp_path, monkeypatch):
    from pipeline import docker
    from pipeline.runlog import EventLog
    from pipeline.sandbox import Sandbox, SandboxError, _BASH_RUN, _BASH_SYNTAX_CHECK
    from pipeline.util import run_bounded
    cfg = Config('p', 'm', tool_output_limit_bytes=128)
    log = EventLog(tmp_path, 'r1')
    sandbox = Sandbox(cfg, 'r1', tmp_path, log, Path('pipeline'))
    def capture(command, **kwargs):
        if _BASH_SYNTAX_CHECK in command:
            return run_bounded([sys.executable, '-I', '-S', '-c', _BASH_SYNTAX_CHECK,
                                'printf "%10000s" x', '1'], **kwargs)
        if _BASH_RUN in command:
            return run_bounded([sys.executable, '-I', '-S', '-c', _BASH_RUN,
                                'printf "%10000s" x', '1', '128'], **kwargs)
        return run_bounded([sys.executable, '-c', 'print("x"*10000)'], **kwargs)
    monkeypatch.setattr(docker, 'run_bounded', capture)
    try:
        result = sandbox.bash('no Docker is called', None, 1)
        assert result['truncated']
        assert len(result['output']) < 200
        with pytest.raises(SandboxError, match='exceeds output limit'):
            sandbox.read_file('/large-file')
    finally:
        log.close()


def test_truncated_replay_prefix_cannot_confirm_solution(tmp_path):
    from pipeline.evaluate import ReplayResult
    root = package(tmp_path / 'r1')
    write_json(root / 'passport.json', {'schema': 'evaluator-pipeline/passport/v2'})
    (root / 'submission/solution.py').write_text('pass')
    write_json(root / 'submission_manifest.json', {
        'present': True, 'health': 'ok', 'files': tree_manifest(root / 'submission'),
    })
    env = Mock()
    env.prepare.return_value = 'http://task'
    env.run.return_value = ReplayResult(0, 'fresh', '', 1, ['python'], stdout_truncated=True)
    env.close.return_value = []
    report = evaluate(root, replay_env_factory=lambda: env, new_flag='fresh')
    assert report['script_check']['verdict'] == 'not_confirmed'
    assert report['script_check']['stdout_truncated']


def test_raw_references_use_physical_lines_after_corruption(tmp_path):
    (tmp_path / 'raw').mkdir()
    record = {'type': 'agent_start'}
    event = {'seq': 1, 'type': 'agent_start', 'data': record, 'raw_line': 2}
    (tmp_path / 'events.jsonl').write_text(json.dumps(event) + '\n')
    (tmp_path / 'raw/rpc.jsonl').write_text('broken\n' + json.dumps(record) + '\n')
    errors = build_data_health(tmp_path, [event])['recording_errors']
    assert len(errors) == 1
    assert errors[0]['line'] == 1


def test_process_capture_drains_while_sending_input():
    from pipeline.util import run_bounded
    result = run_bounded([sys.executable, '-c',
        'import sys; sys.stderr.write("e"*100000); sys.stdout.buffer.write(sys.stdin.buffer.read())'],
        timeout=3, output_limit=256, input_data=b'x' * 200000)
    assert result.returncode == 0
    assert result.stdout == b'x' * 256
    assert result.stdout_truncated and result.stderr_truncated


def test_failed_pi_start_removes_temporary_credentials(tmp_path, monkeypatch):
    from pipeline.pi_rpc import PiSession
    from pipeline.runlog import EventLog
    auth = tmp_path / 'source-auth.json'
    auth.write_text('{}')
    log = EventLog(tmp_path, 'r1')
    session = PiSession(Config('p', 'm', auth_file=str(auth)), log, tmp_path,
                        Path('bridge.ts'), 'url', 'token', 'p', 'm', 'off')
    monkeypatch.setattr(session, '_argv', Mock(side_effect=FileNotFoundError('pi missing')))
    try:
        with pytest.raises(FileNotFoundError):
            session.start()
        temporary = session._agent_dir
        assert temporary.is_dir()
        session.close()
        assert not temporary.exists()
        assert session._stderr_handle.closed
    finally:
        session.close()
        log.close()


def test_pi_close_kills_descendants_after_parent_exit(tmp_path):
    import os
    from pipeline.pi_rpc import PiSession
    from pipeline.runlog import EventLog
    log = EventLog(tmp_path, 'r1')
    session = PiSession(Config('p', 'm'), log, tmp_path, Path('bridge.ts'),
                        'url', 'token', 'p', 'm', 'off')
    proc = subprocess.Popen([sys.executable, '-c',
        'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    session.process = proc
    try:
        proc.wait(timeout=3)
        session.close()
        # The descendant inherited these pipes; EOF proves it did not survive close.
        proc.communicate(timeout=3)
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass
        proc.wait()
        log.close()
