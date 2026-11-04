"""Execute only the observation cell with fake APIs and display, never a run."""
import json
import sys
import types
from pathlib import Path


def test_monitor_refresh_and_escaping(monkeypatch):
    notebook = json.loads(Path('notebooks/remote-duckerz.ipynb').read_text())
    source = next(''.join(c['source']) for c in notebook['cells']
                  if c['cell_type'] == 'code' and 'from html import escape\n' in c['source'])
    displayed = []
    display_module = types.ModuleType('IPython.display')
    display_module.HTML = lambda html: html
    display_module.display = displayed.append
    monkeypatch.setitem(sys.modules, 'IPython.display', display_module)
    batches = [[
        {'seq': 1, 'type': 'message_update', 'data': {'delta': 'hidden-token'}},
        {'seq': 2, 'type': 'tool_execution_start', 'data': {
            'toolName': 'bash', 'args': {'command': '<script>alert(1)</script>'}}},
        {'seq': 3, 'type': 'tool_execution_end', 'data': {
            'toolName': 'bash', 'result': {'content': [{'type': 'text', 'text': 'result-text'}]}}},
        {'seq': 4, 'type': 'message_end', 'data': {
            'message': {'content': [{'type': 'text', 'text': 'hidden-message'}]}}},
        {'seq': 5, 'type': 'stage.collect', 'data': {'detail': 'hidden-stage'}},
        {'seq': 6, 'type': 'tool_exec', 'source': 'sandbox', 'data': {
            'output': 'shell-output', 'exit_code': 124, 'timed_out': True}},
    ], [], []]
    paths = []
    def api(path):
        paths.append(path)
        if '/events?' in path:
            return batches.pop(0)
        return {'status': 'running', 'stage': 'agent.running', 'remaining_seconds': 42}
    scope = {'RUN_ID': 'r1', 'api': api, 'json': json, 'display': displayed.append}
    exec(source, scope)
    assert '42 с' in displayed[-1]
    assert 'hidden-token' not in displayed[-1]
    assert 'hidden-message' not in displayed[-1]
    assert 'hidden-stage' not in displayed[-1]
    assert 'shell-output' in displayed[-1]
    assert 'exit_code: 124' in displayed[-1]
    assert 'timed_out: True' in displayed[-1]
    assert '<script>' not in displayed[-1]
    assert '&lt;script&gt;' in displayed[-1]
    assert 'result-text' in displayed[-1]
    assert '<details>' in displayed[-1]
    exec(source, scope)
    assert paths[-1].endswith('after=6')
    assert 'result-text' in displayed[-1]  # Empty refresh preserves history.
    scope['RUN_ID'] = 'r2'
    exec(source, scope)
    assert paths[-1] == '/runs/r2/events?after=0'
    assert 'result-text' not in displayed[-1]
    assert 'Пока нет действий' in displayed[-1]
