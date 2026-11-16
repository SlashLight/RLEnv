"""Terminal notebook observation from saved/API evidence, no live model."""
import json
import importlib
from pathlib import Path

from pipeline.notebook import Observer, short, summary
from pipeline.config import Config
from pipeline.runner import Runner
from pipeline.runlog import EventCursor, EventLog
from pipeline.util import write_json


def event(seq, kind, data=None, **extra):
    return {"schema": "evaluator-pipeline/event/v2", "run_id": "r1", "seq": seq,
            "ts": "2026-11-14T10:00:00Z", "source": "pi", "type": kind,
            "data": data, "raw_refs": [], "incomplete": False, "truncated": False, **extra}


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
    assert 'Observer(api, RUN_ID, run_dir=RUN_DIR)' in source
    assert 'display(HTML(observer.details_html()))' in source
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
            return {'solution': {'solved': True, 'status': 'confirmed'}}
        return state

    observer = Observer(api, 'r1')
    observer.watch(seconds=0)
    state['status'] = 'done'
    observer.watch(seconds=0)
    output = capsys.readouterr().out
    assert output.count('Агент: Готово') == 1
    assert 'Сейчас: Завершено: done' in output
    assert 'Полное решение: true' in output
    assert '/runs/r1/events?after=1' in paths
    assert not any('/stop' in path for path in paths)


def test_state_exposes_only_display_metadata(tmp_path):
    root = tmp_path / 'r1'
    write_json(root / 'status.json', {'status': 'done'})
    write_json(root / 'passport.json', {'task': {'id': 'ctf'}, 'agent': {'model': 'model'}, 'private': 'secret'})
    state = Runner(Config('p', 'm', runs_dir=tmp_path)).state('r1')
    assert state['viewer'] == {'task': 'ctf', 'model': 'model'}
    assert 'secret' not in str(state)


def test_saved_events_remain_readable_without_report(tmp_path, capsys):
    write_json(tmp_path / 'status.json', {'status': 'running', 'stage': 'agent.running'})
    write_json(tmp_path / 'checks/flag.json',
               {'verdict': 'confirmed', 'predicate': 'exact submitted flag', 'reason': 'match'})
    record = event(1, 'tool_execution_start',
                   {'toolCallId': 'a', 'toolName': 'bash', 'args': {'command': 'curl target'}})
    (tmp_path / 'events.jsonl').write_text(json.dumps(record) + '\n{"seq":2')
    observer = Observer(None, 'r1', run_dir=tmp_path)
    observer.watch(seconds=0)
    output = capsys.readouterr().out
    assert 'Неподтверждённое сохранённое состояние: running' in output
    assert 'Последняя известная активность:' in output and '2026-11-14' in output
    assert 'curl target' in output
    assert 'Окончательный вердикт: отсутствует' in output
    assert 'Проверка flag_check: confirmed' in output
    assert 'Проверка script_check: нет сохранённого результата' in output
    assert 'Неполные свидетельства' in output
    assert 'Полное решение: false' not in output and 'check_error' not in output
    assert observer.after == 1


def test_connection_loss_keeps_last_activity_without_inventing_result(capsys):
    online = True
    records = [event(1, 'message_end',
                     {'message': {'role': 'assistant', 'content': 'Я нашёл флаг'}})]
    def api(path):
        if not online:
            raise ConnectionError('offline')
        if '/events?' in path:
            return records
        return {'status': 'running', 'live_owned': True}
    observer = Observer(api, 'r1')
    observer.watch(seconds=0)
    capsys.readouterr()
    online = False
    observer.watch(seconds=0)
    output = capsys.readouterr().out
    assert 'Связь потеряна' in output
    assert 'Неподтверждённое сохранённое состояние: running' in output
    assert 'Последняя известная активность:' in output and 'message_end' in output
    assert observer.report is None
    assert 'Полное решение: false' not in output


def test_summary_separates_verdict_attempts_usage_and_stop_reason():
    report = {
        'solution': {'solved': None},
        'stop_reason': {'kind': 'usage_unknown'},
        'flag_check': {'verdict': 'confirmed', 'predicate': 'exact flag', 'evidence': ['flag.txt']},
        'script_check': {
            'verdict': 'check_error', 'basis_attempt_id': 'replay-2',
            'attempts': [
                {'attempt_id': 'replay-1', 'verdict': 'check_error', 'reason': 'infra'},
                {'attempt_id': 'replay-2', 'verdict': 'check_error', 'reason': 'infra',
                 'cleanup_errors': ['cleanup unavailable']},
            ],
        },
        'milestones': {'reason': 'no_milestones', 'checks': []},
        'budgets': {'limits': {'token_budget': 200000}},
        'usage': {'total_tokens': 123, 'token_budget': 200000, 'usage_reliable': False,
                  'cost': {'kind': 'unknown', 'value': None}},
    }
    output = '\n'.join(summary({'status': 'done'}, report))
    assert 'Полное решение: null' in output and 'usage_unknown' in output
    assert 'exact flag' in output and 'flag.txt' in output
    assert 'Попытка replay-1' in output and 'Попытка replay-2' in output
    assert 'Итог основан на попытке: replay-2' in output
    assert 'Очистка (отдельно от вердикта)' in output
    assert 'не предусмотрены задачей' in output
    assert '123 / 200000' in output and 'нижняя граница' in output
    assert 'Стоимость: нет данных' in output
    assert 'reward' not in output.lower() and 'score' not in output.lower()
    report['solution']['solved'] = True
    report['usage']['cost'] = {'kind': 'provider_total', 'value': 0, 'currency': 'USD'}
    assert 'Стоимость: 0 USD (provider_total)' in '\n'.join(summary({'status': 'done'}, report))


def test_event_contract_full_disclosure_and_missing_raw(tmp_path):
    raw = {'type': 'message_end', 'message': {
        'role': 'assistant', 'content': '<script>flag</script>' + 'z' * 2000}}
    log = EventLog(tmp_path, 'r1', event_limit_bytes=700)
    line = log.raw_record(raw)
    first = log.publish('pi', 'message_end', raw, raw_line=line)
    second = log.publish('runner', 'stage.collect', {})
    log.close()
    assert first['schema'] == 'evaluator-pipeline/event/v2'
    assert (first['run_id'], first['seq']) == ('r1', 1)
    assert first['raw_refs'] == [{'path': 'raw/rpc.jsonl', 'line': 1}]
    assert first['ts'] and first['source'] == 'pi'
    assert first['truncated'] is True and first['incomplete'] is True
    assert second['seq'] == 2 and second['incomplete'] is False
    observer = Observer(None, 'r1', run_dir=tmp_path)
    observer.update([first, second])
    (tmp_path / 'observations.md').write_text('Notes <script>not proof</script>')
    html = observer.details_html()
    assert '<details>' in html and '<script>' not in html
    assert '&lt;script&gt;flag&lt;/script&gt;' in html and 'z' * 2000 in html
    assert 'raw/rpc.jsonl' in html and 'не независимые доказательства' in html
    assert 'Notes &lt;script&gt;not proof&lt;/script&gt;' in html
    (tmp_path / 'raw/rpc.jsonl').unlink()
    assert 'Исходное свидетельство отсутствует' in observer.details_html()


def test_event_cursor_exposes_corrupt_and_unfinished_records(tmp_path):
    path = tmp_path / 'events.jsonl'
    path.write_text(json.dumps(event(1, 'agent_start')) + '\nbroken\n{"seq":2')
    cursor = EventCursor()
    assert [item['seq'] for item in cursor.read(tmp_path, 0)] == [1]
    assert cursor.recording_errors and cursor.incomplete_tail is True
    with path.open('a') as stream:
        stream.write(',"type":"agent_end"}\n')
    assert [item['seq'] for item in cursor.read(tmp_path, 1)] == [2]
    assert cursor.incomplete_tail is False


def test_finished_state_without_report_does_not_invent_infrastructure_failure(capsys):
    def api(path):
        if '/events?' in path:
            return []
        if path.endswith('/report'):
            raise RuntimeError('report not ready')
        return {'status': 'done', 'stop_reason': {'kind': 'submitted'}}
    Observer(api, 'r1').watch(seconds=0)
    output = capsys.readouterr().out
    assert 'Окончательный вердикт: отсутствует' in output
    assert 'submitted' in output
    assert 'Полное решение: false' not in output and 'infrastructure_error' not in output


def test_remote_event_shape_is_unchanged(tmp_path):
    log = EventLog(tmp_path, 'remote', local_contract=False)
    record = log.publish('runner', 'stage.prepare', {})
    log.close()
    assert set(record) == {'run_id', 'seq', 'ts', 'source', 'type', 'data'}
