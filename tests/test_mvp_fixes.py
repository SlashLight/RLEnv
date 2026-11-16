"""Regression coverage for MVP reports and terminal outcome precedence."""
import json
import threading
import time
from pathlib import Path

import pytest

from pipeline.config import Config
from pipeline.evaluate import evaluate, render_markdown
from pipeline.pi_rpc import PiSession
from pipeline.runner import Runner
from pipeline.runlog import EventLog
from pipeline.server import Api
from pipeline.util import write_json


@pytest.mark.parametrize('action', ['report', 'reevaluate'])
def test_notebook_report_works_from_different_cwd(tmp_path, monkeypatch, action):
    notebook = json.loads(Path('notebooks/evaluator-pipeline.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if f'/{{RUN_ID}}/{action}\"' in ''.join(cell['source']))
    monkeypatch.chdir(tmp_path)
    root = Path('runs/r1')
    write_json(root / 'passport.json', {'run_id': 'r1', 'schema': 'evaluator-pipeline/passport/v2'})
    write_json(root / 'status.json', {'status': 'done'})
    report = evaluate(root, execute_replay=False)
    runner = Runner(Config('p', 'm', runs_dir=(tmp_path / 'runs')))
    monkeypatch.setattr(runner, 'reassess', lambda run_id: report)
    api = Api(runner)
    client = tmp_path / 'notebooks'
    client.mkdir()
    monkeypatch.chdir(client)
    rendered = []

    def request(path, method='GET', body=None):
        code, payload = api.get(path, {}) if method == 'GET' else api.post(path, body)
        assert code == 200
        return payload

    exec(compile(source, 'notebook-report', 'exec'), {
        'api': request, 'RUN_ID': 'r1', 'Path': Path,
        'Markdown': lambda text: text, 'display': rendered.append,
    })
    assert rendered == [render_markdown(report)]


@pytest.mark.parametrize('condition, expected', [
    ('model_error', 'error'), ('user_stop', 'stopped'), ('settled', 'settled'),
])
def test_terminal_outcome_precedence(tmp_path, condition, expected):
    log = EventLog(tmp_path, 'r1')
    try:
        session = PiSession(Config('p', 'm'), log, tmp_path, Path('bridge.ts'),
                            'http://gateway', 'token', 'p', 'm', 'off', local=False)
        if condition == 'model_error':
            session._inspect({'type': 'message_end', 'message': {
                'role': 'assistant', 'stopReason': 'error', 'errorMessage': 'API failed',
            }})
        session._inspect({'type': 'agent_settled'})
        stop = threading.Event()
        if condition == 'user_stop':
            stop.set()
        assert session.wait_settled(time.monotonic() + 1, stop) == expected
    finally:
        log.close()
