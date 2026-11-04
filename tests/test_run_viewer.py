"""Terminal notebook observation from saved/API evidence, no live model."""
import json
import importlib
from pathlib import Path

from pipeline.notebook import Observer, short
from pipeline.config import Config
from pipeline.runner import Runner
from pipeline.util import write_json


def event(seq, kind, data=None, **extra):
    return {"seq": seq, "type": kind, "data": data, **extra}


def test_notebook_connection_reloads_cached_viewer():
    import pipeline.notebook as module

    notebook = json.loads(Path('notebooks/evaluator-pipeline.ipynb').read_text())
    source = ''.join(notebook['cells'][1]['source']).split("BASE =", 1)[0]
    module.__dict__.pop('Observer')
    try:
        scope = {}
        exec(compile(source, 'notebook-connect', 'exec'), scope)
        assert scope['Observer'] is module.Observer
    finally:
        importlib.reload(module)


def test_notebook_observation_uses_terminal_view_not_raw_event_dump():
    notebook = json.loads(Path('notebooks/evaluator-pipeline.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'observer.watch' in ''.join(cell['source']))
    assert 'Observer(api, RUN_ID)' in source
    assert '/events?after=0' not in source
    assert 'print(json.dumps' not in source
    assert any("/stop'" in ''.join(c['source']) for c in notebook['cells'])
    assert all(not c.get('outputs') for c in notebook['cells'])
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            compile(''.join(cell['source']), 'notebook', 'exec')


def test_current_action_and_completed_steps_without_transport_noise():
    observer = Observer(None, 'r1')
    assert observer.update([
        event(1, 'message_start', {'message': {'role': 'assistant'}}),
        event(2, 'message_update', {'assistantMessageEvent': {'type': 'text_delta', 'delta': 'Проверяю страницу'}}),
        event(3, 'tool_execution_start', {'toolCallId': 'a', 'toolName': 'bash',
                                          'args': {'command': 'curl http://service'}}),
        event(4, 'tool_exec', {'output': 'duplicate'}),
    ]) == []
    assert observer.current == 'Выполняется bash: curl http://service'
    steps = observer.update([event(5, 'tool_execution_end', {'toolCallId': 'a', 'isError': False,
                                                              'result': {'content': [{'type': 'text', 'text': 'HTTP 200'}]}}),
                             event(6, 'message_end', {'message': {'role': 'assistant', 'content': [
                                 {'type': 'text', 'text': 'Проверяю страницу'},
                                 {'type': 'thinking', 'thinking': 'hidden'},
                             ]}})])
    assert len(steps) == 2
    assert 'HTTP 200' in steps[0] and 'Проверяю страницу' in steps[1]
    assert 'hidden' not in str(steps) and 'duplicate' not in str(steps)
    assert observer.after == 6


def test_streaming_text_is_current_but_not_raw_json():
    observer = Observer(None, 'r1')
    assert observer.update([event(1, 'message_start', {'message': {'role': 'assistant'}}),
                            event(2, 'message_update', {'assistantMessageEvent': {'type': 'text_delta', 'delta': 'Сейчас '}}),
                            event(3, 'message_update', {'assistantMessageEvent': {'type': 'text_delta', 'delta': 'пишу скрипт'}})]) == []
    assert observer.current == 'Агент: Сейчас пишу скрипт'
    observer.update([event(4, 'message_update', {'assistantMessageEvent': {'type': 'toolcall_start', 'toolName': 'write'}})])
    assert observer.current == 'Агент готовит вызов write'
    assert 'usage' not in observer.current


def test_parallel_calls_keep_current_action_after_one_finishes():
    observer = Observer(None, 'r1')
    observer.update([event(1, 'tool_execution_start', {'toolCallId': 'a', 'toolName': 'read', 'args': {'path': '/a'}}),
                     event(2, 'tool_execution_start', {'toolCallId': 'b', 'toolName': 'read', 'args': {'path': '/b'}}),
                     event(3, 'tool_execution_end', {'toolCallId': 'a', 'result': {}})])
    assert observer.current == 'Выполняется read: /b'


def test_timeout_missing_result_and_control_chars():
    observer = Observer(None, 'r1')
    observer.update([event(1, 'tool_execution_start', {'toolCallId': 'a', 'toolName': 'bash',
                                                        'args': {'command': 'sleep 20'}})])
    assert 'Выполняется' in observer.current
    steps = observer.update([event(2, 'tool_execution_end', {'toolCallId': 'a', 'isError': True,
        'result': {'content': [{'type': 'text', 'text': '\x1b[31merror\x1b[0m'}],
                   'structuredContent': {'timed_out': True, 'truncated': True}}})])
    assert 'тайм-аут' in steps[0] and 'вывод обрезан' in steps[0]
    assert '\x1b' not in steps[0]
    assert short('x' * 500).endswith('…')


def test_watch_show_current_action_and_summary(capsys):
    calls = []
    events = [event(1, 'tool_execution_start', {'toolCallId': 'a', 'toolName': 'write',
                                                  'args': {'path': '/workspace/solution.py', 'content': 'x' * 1000}})]

    def api(path):
        calls.append(path)
        if '/events?' in path:
            return events
        return {'status': 'running', 'stage': 'agent.running', 'viewer': {'task': 'ctf', 'model': 'm'}}

    observer = Observer(api, 'r1')
    observer.watch(seconds=0)
    output = capsys.readouterr().out
    assert 'Сейчас: Выполняется write: /workspace/solution.py' in output
    assert 'Последние шаги:' in output
    assert 'run_id' not in output and 'toolCallId' not in output and 'x' * 100 not in output
    assert calls == ['/runs/r1', '/runs/r1/events?after=0']


def test_reconnect_after_bounded_watch_and_final_report(capsys):
    events = [event(1, 'message_end', {'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Готово'}]}})]
    state = {'status': 'running'}
    paths = []

    def api(path):
        paths.append(path)
        if '/events?' in path:
            return [e for e in events if e['seq'] > int(path.rsplit('=', 1)[-1])]
        if path.endswith('/report'):
            return {'solution': {'status': 'solved'}}
        return state

    observer = Observer(api, 'r1')
    observer.watch(seconds=0)
    state['status'] = 'done'
    observer.watch(seconds=0)
    output = capsys.readouterr().out
    assert output.count('Агент: Готово') == 1
    assert 'Сейчас: Завершено: done' in output
    assert 'Итог: solved' in output
    assert '/runs/r1/events?after=1' in paths
    assert not any('/stop' in path for path in paths)


def test_state_exposes_only_display_metadata(tmp_path):
    root = tmp_path / 'r1'
    write_json(root / 'status.json', {'status': 'done'})
    write_json(root / 'passport.json', {'task': {'id': 'ctf'}, 'agent': {'model': 'model'}, 'private': 'secret'})
    state = Runner(Config('p', 'm', runs_dir=tmp_path)).state('r1')
    assert state['viewer'] == {'task': 'ctf', 'model': 'model'}
    assert 'secret' not in str(state)
