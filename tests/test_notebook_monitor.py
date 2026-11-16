"""Execute only the observation cell with fake APIs and display, never a run."""
import json
from pathlib import Path




def test_local_observation_cell_discloses_saved_checks_and_escaped_details(tmp_path, capsys):
    from pipeline.notebook import Observer

    notebook = json.loads(Path('notebooks/evaluator-pipeline.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells']
                  if cell['cell_type'] == 'code' and 'observer.watch' in ''.join(cell['source']))
    displayed = []
    record = {'schema': 'evaluator-pipeline/event/v2', 'run_id': 'r1', 'seq': 1,
              'ts': '2026-11-14T10:00:00Z', 'source': 'pi', 'type': 'message_end',
              'data': {'message': {'role': 'assistant',
                                   'content': '<script>claim</script>' + 'x' * 1000}},
              'incomplete': True, 'truncated': True, 'raw_refs': []}
    def api(path):
        if '/events?' in path:
            return [record]
        assert not path.endswith('/report')
        return {'status': 'running', 'live_owned': False,
                'available_checks': {
                    'flag': {'verdict': 'confirmed', 'predicate': 'exact flag', 'reason': 'match'},
                    'replay': {'verdict': 'not_evaluated', 'reason': 'not_started'},
                },
                'budgets': {'token_budget': 200000},
                'usage': {'total_tokens': 10, 'token_budget': 200000,
                          'usage_reliable': False, 'cost': {'kind': 'unknown', 'value': None}}}
    scope = {'Observer': Observer, 'ROOT': tmp_path, 'RUN_ID': 'r1', 'api': api,
             'display': displayed.append, 'HTML': lambda value: value}
    exec(source, scope)
    output = capsys.readouterr().out
    assert 'Проверка flag_check: confirmed' in output
    assert 'Проверка script_check: not_evaluated' in output
    assert 'Окончательный вердикт: отсутствует' in output
    assert 'Неподтверждённое сохранённое состояние: running' in output
    assert 'Стоимость: нет данных' in output
    assert '<details>' in displayed[-1]
    assert '<script>' not in displayed[-1] and '&lt;script&gt;claim&lt;/script&gt;' in displayed[-1]
    assert 'x' * 1000 in displayed[-1]
    assert 'не независимые доказательства' in displayed[-1]
