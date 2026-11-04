"""Trusted syntax receipts; Docker is mocked and parser probes use controlled fixtures."""
import base64
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from pipeline import docker
from pipeline.config import Config
from pipeline.gateway import Gateway
from pipeline.replay import (
    DockerReplayEnvironment, _PYTHON_SYNTAX_AND_RUN, _SYNTAX_HEADER_LIMIT, _SYNTAX_PREFIX,
)
from pipeline.sandbox import Sandbox, SandboxError, _BASH_RUN, _BASH_SYNTAX_CHECK
from pipeline.util import ProcessResult


def sandbox(tmp_path):
    return Sandbox(Config('p', 'm'), 'syntax', tmp_path, Mock(), Path('pipeline'))


@pytest.mark.parametrize('command, verdict', [
    ('echo ok', 'valid'),
    ('if then', 'invalid'),
    ('missing_program', 'valid'),
    ('printf "syntax error near unexpected token"; exit 2', 'valid'),
    ('eval "if then"', 'valid'),  # only the top-level command is parsed
])
def test_bash_parser_checks_source_not_execution(tmp_path, command, verdict):
    marker = tmp_path / 'executed'
    source = f'exec 2>&1\ntouch {marker}; {command}'
    proc = subprocess.run(
        [sys.executable, '-I', '-S', '-c', _BASH_SYNTAX_CHECK, source, '2'],
        capture_output=True, timeout=5,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)['verdict'] == verdict
    assert not marker.exists()


def test_bash_checker_ignores_bash_env(tmp_path, monkeypatch):
    startup = tmp_path / 'startup'
    marker = tmp_path / 'executed'
    startup.write_text(f'touch {marker}\n')
    monkeypatch.setenv('BASH_ENV', str(startup))
    proc = subprocess.run(
        [sys.executable, '-I', '-S', '-c', _BASH_SYNTAX_CHECK, 'echo ok', '2'],
        capture_output=True, timeout=5,
    )
    assert json.loads(proc.stdout)['verdict'] == 'valid'
    assert not marker.exists()


@pytest.mark.parametrize('receipt', [
    ProcessResult(0, b'{"verdict":"valid","parser_exit_code":0}', b''),
    ProcessResult(0, b'{"verdict":"invalid","parser_exit_code":2}', b''),
    ProcessResult(127, b'', b'no python'),
    ProcessResult(124, b'', b'timed out'),
    ProcessResult(2, b'', b'syntax error'),  # Docker/launcher error, not parser evidence
    ProcessResult(0, b'not json', b''),
    ProcessResult(0, b'{"verdict":"invalid"}', b'', True),
    docker.DockerError('daemon unavailable'),
])
def test_sandbox_receipt_and_unchanged_execution(tmp_path, receipt):
    box = sandbox(tmp_path)
    actual = ProcessResult(0, json.dumps({
        'exit_code': 127, 'timed_out': False, 'truncated': False,
        'output_b64': base64.b64encode(b'missing tool').decode(),
    }).encode(), b'')
    box._exec = Mock(side_effect=[receipt, actual])
    if (isinstance(receipt, Exception) or receipt.returncode
            or receipt.stdout_truncated or receipt.stdout == b'not json'):
        with pytest.raises(SandboxError, match='checker unavailable'):
            box.bash('missing tool', '/workspace', 2)
        assert box._exec.call_count == 1
        return
    result = box.bash('missing tool', '/workspace', 2)
    assert result['syntax_check']['verdict'] == json.loads(receipt.stdout)['verdict']
    assert result['syntax_check']['evidence_version'] == 3
    if result['syntax_check']['verdict'] == 'invalid':
        assert box._exec.call_count == 1
        assert result['exit_code'] == 2
        return
    checker, execution = box._exec.call_args_list
    executable_command = 'exec 2>&1\nmissing tool'
    assert checker.args[0][-2] == executable_command == execution.args[0][-3]
    assert _BASH_RUN in execution.args[0]
    assert '["/bin/bash", "--noprofile", "--norc", "-O", "extglob", "-c", sys.argv[1]]' in _BASH_RUN
    assert '-I' in checker.args[0] and '-S' in checker.args[0]
    assert result['syntax_check']['source_sha256'] == hashlib.sha256(executable_command.encode()).hexdigest()
    assert result['exit_code'] == 127 and result['output'] == 'missing tool'


def local_supervised_box(tmp_path):
    box = sandbox(tmp_path)

    def controlled_exec(argv, **kwargs):
        # Execute only our trusted supervisors with test-owned commands, not Docker.
        code_index = argv.index('-c')
        proc = subprocess.run(
            [sys.executable, '-I', '-S', '-c', *argv[code_index + 1:]],
            cwd=tmp_path, capture_output=True, timeout=kwargs['timeout'],
        )
        return ProcessResult(proc.returncode, proc.stdout, proc.stderr)

    box._exec = Mock(side_effect=controlled_exec)
    return box


@pytest.mark.parametrize('command, timed_out, code', [
    ('exit 124', False, 124),
    ('sleep 10', True, 124),
    ('printf \'{"timed_out":true,"exit_code":124}\\n\'; exit 2', False, 2),
    ('printf "syntax error near unexpected token\\n" >&2; exit 2', False, 2),
])
def test_supervised_timeout_is_not_child_exit_or_output(tmp_path, command, timed_out, code):
    result = local_supervised_box(tmp_path).bash(command, str(tmp_path), .15)
    assert result['exit_code'] == code
    assert result['timed_out'] is timed_out
    assert result['syntax_check']['verdict'] != 'invalid'


@pytest.mark.parametrize('tail, verdict', [('', 'valid'), ('\nexit 2', 'valid')])
def test_extglob_static_failure_is_not_penalized(tmp_path, tail, verdict):
    command = 'shopt -s extglob\nprintf "%s\\n" +(x)' + tail
    result = local_supervised_box(tmp_path).bash(command, str(tmp_path), 2)
    assert result['syntax_check']['verdict'] == verdict


def test_invalid_input_is_rejected_before_any_side_effect(tmp_path):
    box = local_supervised_box(tmp_path)
    result = box.bash('touch marker\nif then', str(tmp_path), 2)
    assert result['exit_code'] == 2
    assert result['syntax_check']['verdict'] == 'invalid'
    assert not (tmp_path / 'marker').exists()
    assert box._exec.call_count == 1


def test_supervisor_bounds_output(tmp_path):
    box = local_supervised_box(tmp_path)
    box.cfg.tool_output_limit_bytes = 128
    result = box.bash('printf "%10000s" x', str(tmp_path), 2)
    assert result['truncated']
    assert len(result['output']) < 200
    assert not result['timed_out']


@pytest.mark.parametrize('wire', [
    ProcessResult(124, b'', b''),
    ProcessResult(0, b'{"timed_out":true}', b''),
    ProcessResult(0, b'not json', b''),
    ProcessResult(0, b'{"exit_code":0,"timed_out":false,"truncated":false,"output_b64":"!"}', b''),
])
def test_supervisor_failures_are_not_child_timeouts(tmp_path, wire):
    box = sandbox(tmp_path)
    box._bash_syntax = Mock(return_value={'verdict': 'valid'})
    box._exec = Mock(return_value=wire)
    with pytest.raises(SandboxError, match='supervisor'):
        box.bash('exit 0', None, 1)


def test_supervisor_stdout_cannot_be_opened_through_proc(tmp_path):
    command = 'printf FORGED > /proc/$PPID/fd/1; printf genuine'
    result = local_supervised_box(tmp_path).bash(command, str(tmp_path), 2)
    assert result['exit_code'] == 0
    assert 'genuine' in result['output']
    assert 'Permission denied' in result['output']


def test_gateway_propagates_receipt_and_supports_old_sandbox():
    gateway = object.__new__(Gateway)
    gateway.sandbox = Mock()
    result = dict(exit_code=0, timed_out=False, output='ok', truncated=False, duration_ms=1)
    gateway.sandbox.bash.return_value = result
    payload, audit = gateway.handle('bash', {'command': 'echo ok'})
    assert 'syntax_check' not in payload and 'syntax_check' not in audit
    result['syntax_check'] = {'verdict': 'valid'}
    payload, audit = gateway.handle('bash', {'command': 'echo ok'})
    assert payload['syntax_check'] == audit['syntax_check'] == result['syntax_check']


@pytest.mark.parametrize('source, verdict', [
    ('raise RuntimeError("must not execute")', 'valid'),
    ('if :', 'invalid'),
    ('import missing_dependency', 'valid'),
])
def test_python_parser_does_not_execute_or_import_submission(tmp_path, source, verdict):
    (tmp_path / 'solution.py').write_text(source)
    for name in ('json.py', 'hashlib.py', 'sitecustomize.py'):
        (tmp_path / name).write_text('raise RuntimeError("submission import")')
    # Replace only the final execution; the trusted checker itself is unchanged.
    code = 'import os\nos.execv = lambda *args: None\n' + _PYTHON_SYNTAX_AND_RUN
    proc = subprocess.run([sys.executable, '-I', '-S', '-c', code, 'http://task'],
                          cwd=tmp_path, capture_output=True, timeout=5)
    assert proc.returncode == 0, proc.stderr
    receipt = json.loads(proc.stdout.decode().removeprefix(_SYNTAX_PREFIX))
    assert receipt['verdict'] == verdict
    assert receipt['source_sha256'] == hashlib.sha256(source.encode()).hexdigest()


def test_python_receipt_precedes_normal_replay_execution(tmp_path):
    (tmp_path / 'solution.py').write_text('import sys, helper\nprint(helper.value, sys.argv[1])')
    (tmp_path / 'helper.py').write_text('value = "flag"')
    proc = subprocess.run(
        [sys.executable, '-I', '-S', '-c', _PYTHON_SYNTAX_AND_RUN, 'http://task'],
        cwd=tmp_path, capture_output=True, timeout=5,
    )
    assert proc.returncode == 0, proc.stderr
    header, output = proc.stdout.decode().split('\n', 1)
    assert json.loads(header.removeprefix(_SYNTAX_PREFIX))['verdict'] == 'valid'
    assert output == 'flag http://task\n'


def replay(tmp_path, monkeypatch, output, *, limit=1024, code=0, timeout=False):
    submission = tmp_path / 'submission'
    submission.mkdir()
    (submission / 'solution.py').write_text('pass')
    env = DockerReplayEnvironment(Config('p', 'm', tool_output_limit_bytes=limit), Mock(), tmp_path)
    env.tools_image = 'tools'
    failure = subprocess.TimeoutExpired(['docker'], 1, output=output, stderr=b'')
    run = Mock(side_effect=failure) if timeout else Mock(return_value=ProcessResult(code, output, b''))
    monkeypatch.setattr('pipeline.replay.run_bounded', run)
    state = {'Status': 'exited', 'ExitCode': code, 'Error': '', 'OOMKilled': False}
    inspect = Mock(return_value=ProcessResult(0, json.dumps(state).encode(), b''))
    monkeypatch.setattr(docker, 'docker', inspect)
    monkeypatch.setattr(docker, 'remove_container', Mock())
    return env.run(submission, 'http://task', 1), run


def receipt(verdict='valid'):
    return (_SYNTAX_PREFIX + json.dumps({'verdict': verdict, 'language': 'python'}) + '\n').encode()


def test_replay_strips_only_trusted_first_receipt(tmp_path, monkeypatch):
    forged = receipt('invalid')
    result, run = replay(tmp_path, monkeypatch, receipt() + b'flag\n' + forged)
    assert result.syntax_check['verdict'] == 'valid'
    assert result.stdout == 'flag\n' + forged.decode()
    command = run.call_args.args[0]
    assert command[-6:] == ['/usr/local/bin/python', '-I', '-S', '-c', _PYTHON_SYNTAX_AND_RUN, 'http://task']
    for option in ('--read-only', '--memory', '--cpus', '--pids-limit', '--cap-drop'):
        assert option in command
    assert run.call_args.kwargs['timeout'] == 21


def test_replay_preserves_output_limit(tmp_path, monkeypatch):
    result, run = replay(tmp_path, monkeypatch, receipt() + b'123456789', limit=5)
    assert result.stdout == '12345' and result.stdout_truncated
    assert run.call_args.kwargs['output_limit'] == 5 + _SYNTAX_HEADER_LIMIT


@pytest.mark.parametrize('output, verdict', [(b'flag', 'check_error'),
                                            (receipt() + b'partial', 'valid')])
def test_replay_timeout_does_not_invent_syntax_failure(tmp_path, monkeypatch, output, verdict):
    result, _ = replay(tmp_path, monkeypatch, output, timeout=True)
    assert result.timed_out
    assert result.syntax_check['verdict'] == verdict


def test_old_replay_mock_output_is_preserved(tmp_path, monkeypatch):
    result, _ = replay(tmp_path, monkeypatch, b'flag')
    assert result.stdout == 'flag'
    assert result.syntax_check['verdict'] == 'check_error'
