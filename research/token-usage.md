# Usage и мягкий токенный бюджет: pi RPC → OpenRouter

Исследование для [вопроса об usage](https://github.com/SlashLight/RLEnv/issues/8) в [карте спецификации](https://github.com/SlashLight/RLEnv/issues/1). Контекст — [согласованное решение об исходах и бюджетах](https://github.com/SlashLight/RLEnv/issues/3#issuecomment-6879513068): 200 000 входных + выходных токенов всех обращений, включая кешированные входные; проверка после ответа, следующий запрос запрещён при достижении лимита; неизвестный usage — инфраструктурная остановка исследования, сохранение и проверка доступной сдачи. Денежный лимит не вводится.

## Краткий вывод

Публичный контракт OpenRouter предоставляет нужные счётчики. В pi они доступны через RPC, но **одного текущего RPC недостаточно, чтобы доказать полноту usage**: адаптер нормализует отсутствие полей в нули и не передаёт признак получения полного provider usage. Наличие объекта `usage`, положительная сумма и равенство его компонентов — не доказательство полноты. Нужны подтверждение полноты на границе провайдера или независимая сверка; до этого строгая гарантия бюджета не подтверждена.

Правильная сумма для нормализованного pi: `input + cacheRead + cacheWrite + output`, один раз на обращение. Для исходного OpenRouter: `prompt_tokens + completion_tokens`. Не добавлять кеш или reasoning повторно. Учесть также суммаризацию; не складывать повторные представления одного ответа и накопительные snapshots.

Это исследование, **не реализация и не решение о закрытии тикета**.

## Проверенный локальный стек и границы

- Исходный репозиторий: commit [`e9e2d65`](https://github.com/SlashLight/RLEnv/tree/e9e2d65); до работы `main` имел только untracked пользовательские файлы, включая `research/`, `AGENTS.md`, `CONTEXT.md`, `TRACKER.md`, `rl-environment.html`. Их не переносили и не меняли. Записка создаётся в отдельном worktree на `research/token-usage`.
- Из локального `config/pipeline.json` прочитана только разрешённая проекция: provider `openrouter`, model `deepseek/deepseek-v4.1-flash-1017`, thinking `max`. Конфиг с credentials не опубликован. Это настроенный провайдер pipeline, а не предположение по модели текущей исследовательской сессии. Реальный завершённый прогон и выбранный OpenRouter upstream не проверялись.
- `which pi` разрешается в установленный pi **1.5.2**. Прочитаны локальные README и docs `rpc.md`, `providers.md`, `models.md`, `message-types.md`, `json.md`, relevant `rpc-commands.md`, `compaction.md`. Публичный исходный код проверен по tag v1.5.2, commit [`b4afdddf44e059c67ccf5097e5317d913521b470`](https://github.com/earendil-works/pi/tree/b4afdddf44e059c67ccf5097e5317d913521b470). Это сверка версии, не доказательство побитовой воспроизводимости установленного бинарника.
- [`pipeline/pi_rpc.py`](https://github.com/SlashLight/RLEnv/blob/e9e2d65/pipeline/pi_rpc.py) запускает изолированный профиль, `--mode rpc --no-session --offline`, фиксирует provider/model, отключает auto retry, **включает auto compaction**. Обрабатывает terminal `message_end`, но usage пока не агрегирует и токенный лимит не применяет. `--offline` не означает отсутствие запросов модели: это не подтверждение отказа от сетевого inference.
- [`pipeline/runlog.py`](https://github.com/SlashLight/RLEnv/blob/e9e2d65/pipeline/runlog.py) сохраняет исходные **RPC-записи**, не HTTP/SSE-чанки OpenRouter. `raw/rpc.jsonl` не восстанавливает поля, потерянные до RPC; нормализованные events могут быть усечены. Приватные `runs/` не открывались.

## Первичный публичный контракт OpenRouter

- [Usage accounting](https://openrouter.ai/docs/cookbook/administration/usage-accounting) перечисляет prompt/completion/total, cache read/write, reasoning и cost. Usage включается автоматически; параметры `usage.include` и `stream_options.include_usage` объявлены deprecated и не влияют на поведение. Адаптер pi всё ещё выставляет include_usage, но это не свидетельство фактического получения accounting.
- [Streaming](https://openrouter.ai/docs/api_reference/streaming) описывает отдельный usage chunk непосредственно перед `[DONE]`, с повторным finish_reason и без контента. Нельзя прекращать чтение при первом finish_reason или считать повторный finish_reason новым ответом. [Overview](https://openrouter.ai/docs/api_reference/overview): response token counts используют **native tokenizer модели**, не универсальный нормализованный tokenizer. Здесь «нормализованный pi» означает форму/разбиение полей, а не ретокенизацию.
- [Prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching): cached_tokens — чтения кеша, cache_write_tokens — записи; usage accounting обещает write split только моделям с explicit caching и cache-write pricing. [Reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens): reasoning уже входит в выходные токены, даже если скрыт от пользователя. Отсутствующий optional split не доказывает отсутствие кеша или reasoning.
- Streaming cancellation останавливает processing/billing только у поддерживающего это upstream. OpenRouter может маршрутизировать модель разным провайдерам; префикс `deepseek/` не определяет upstream. [Errors](https://openrouter.ai/docs/api-reference/errors): ошибка возможна внутри HTTP-200 SSE; пустой видимый ответ не означает нулевой расход. Гарантия доставки terminal usage при разрыве соединения в проверенных источниках не найдена.
- [Generation metadata](https://openrouter.ai/docs/api/api-reference/generations/get-request-&-usage-metadata-for-a-generation): authenticated `GET /api/v1/generation?id=...` даёт provider_name, model, cancelled, total_cost и токенные счётчики. [OpenAPI](https://openrouter.ai/openapi.json) допускает null у `native_tokens_prompt`, `native_tokens_completion`, `native_tokens_reasoning`, `native_tokens_cached`; native поля описаны как сообщённые провайдером. Также nullable `tokens_prompt`/`tokens_completion`, но актуальная схема **не называет их normalized**: нельзя без доказательств подменять ими native accounting. Успешное восстановление каждого aborted запроса не гарантировано проверенными источниками. `X-Generation-Id` помогает корреляции на HTTP-границе; его доступность в текущем Python RPC-профиле не проверена.

## Какие данные доступны

| Слой | Данные | Ограничение |
|---|---|---|
| OpenRouter chat completion | `prompt_tokens`, `completion_tokens`, `total_tokens`; детали cache read/write, reasoning; billing fields | Получение и корректность в конкретном живом прогоне не проверены |
| pi `message_end.message.usage` | `input`, `output`, `cacheRead`, `cacheWrite`, `totalTokens`, optional `reasoning`, `cacheWrite1h`, `cost` | Нормализованные, не исходные provider поля; отсутствующие числа могут стать нулями |
| pi `message_update.usage` | Последний накопительный usage ответа | Не дельта; может оставаться нулевым до завершения |
| pi `compaction_end.result.usage` | Usage суммаризации, потенциально нескольких вызовов | Может отсутствовать; при ошибке/abort нет результата |
| RPC `get_session_stats` | Накопительные токены и cost по session, включая tools/summaries | Сверка, не дополнительный расход; `contextUsage` — не бюджет и не полный расход |

Определения и форма событий: [message-types](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/coding-agent/docs/message-types.md), [json stream](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/coding-agent/docs/json.md), [RPC commands](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/coding-agent/docs/rpc-commands.md).

`responseId`/`responseModel` при наличии помогают сверять поколение и фактически названную модель; они не являются гарантированно доступными идентификаторами всех внутренних запросов. `message_end` — authoritative final message; `turn_end.message`, `agent_end.messages`, `get_messages`, session entries могут повторять тот же ответ. `agent_settled` означает прекращение автоматической работы, не границу каждого отдельного модельного запроса ([RPC](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/coding-agent/docs/rpc.md)).

## Формула без двойного учёта

[Адаптер openai-completions](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/ai/src/api/openai-completions.ts#L1514-L1559) преобразует ответ так:

```text
P = raw.prompt_tokens
R = raw.prompt_tokens_details.cached_tokens  # с fallback для иных compatible API
W = raw.prompt_tokens_details.cache_write_tokens
O = raw.completion_tokens
pi.input = max(0, P - R - W)
pi.cacheRead = R
pi.cacheWrite = W
pi.output = O
pi.totalTokens = pi.input + R + W + O
```

При корректном `0 <= R + W <= P` получается `pi.totalTokens = P + O`. `reasoning` — подмножество `output`; `cacheWrite1h` — подмножество `cacheWrite`. Не прибавлять ни их, ни `totalTokens` поверх компонентов. Адаптер вычисляет `totalTokens` сам, а не сохраняет исходный `raw.total_tokens`: его равенство компонентам не является независимой сверкой. `max(0, …)` может скрыть неконсистентный исходный split; проверка `R+W<=P` требует исходных полей.

Пример: `P=1000, R=600, W=100, O=200` → pi `input=300`, сумма **1200**, не 1800/1900. При `reasoning=50` остаётся **1200**. Два разных запроса с тем же кешированным префиксом учитываются оба: бюджет считает обработанные токены обращений, не уникальный текст.

Агрегировать один final usage на каждое обращение, включая ответы с tool calls, доступный usage неудачных/прерванных обращений и все обращения суммаризации. `get_session_stats` использовать для сверки с тем же охватом, не прибавлять к журналу событий. Текущий sandbox bridge разрешает только read/write/edit/bash: вложенные модельные вызовы tools не являются установленной возможностью этого профиля; при их будущем появлении optional tool usage также требует учёта без дублирования ([usage-totals.ts](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/coding-agent/src/core/usage-totals.ts)).

### Особая проблема compaction

Pi может вызвать модель до следующего assistant turn для суммаризации и выполнить несколько summary-вызовов в одной compaction; [документация compaction](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/coding-agent/docs/compaction.md) описывает split user-message spans с двумя summaries и автоматическое overflow recovery. Поэтому проверка только после `agent_settled` слишком поздняя; только assistant `message_end` тоже не покрывает все запросы.

Целевой контроль должен запрещать **любой следующий модельный запрос**, включая внутренний summary, после достижения бюджета/неизвестного usage предыдущего. `compaction_end.result.usage` полезен для общего итога, но агрегат после нескольких summary-вызовов сам по себе не доказывает соблюдение запрета между ними. Нужна проверка/точка остановки на границе каждого model call; возможность её обеспечить текущим неизменённым Python RPC-клиентом не доказана. Отключение compaction было бы изменением поведения, а не незаметным исправлением формулы.

## Отсутствие и неполнота usage

[Код адаптера](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/ai/src/api/openai-completions.ts#L315-L337) заранее создаёт все нули. Он заменяет usage при получении `chunk.usage` либо `choice.usage`; parser применяет `|| 0`/`?? 0` к отсутствующим полям. Признака «исходный usage получен и содержит обязательные поля» в `Usage` нет. Например, исходные `{completion_tokens:200}` и `{prompt_tokens:0,completion_tokens:200}` дают одинаковый токенный результат. Положительный `totalTokens` не устраняет неоднозначность.

Различать:

1. **Подтверждён полный расход**: есть terminal provider accounting и проверены обязательные `prompt_tokens`, `completion_tokens` (и независимый `total_tokens`, когда доступен), числовая конечность/неотрицательность/целочисленность, согласованность total и splits; необязательный cache split не нужен для общей суммы, если полные P/O подтверждены.
2. **Известна только часть**: сохранять подтверждённое значение как частичный расход/нижнюю границу, не выдавать его за полный итог.
3. **Нет подтверждения**: объект из нулей, missing/null обязательных полей, разрыв потока, незавершённый/неверифицированный запрос, неконсистентные числа — unknown, не 0. Нулевой output возможен, поэтому запрещать любое нулевое поле нельзя.

RPC позволяет заметить отсутствующий/некорректный объект, stopReason error/aborted, protocol error/exit, пропавший terminal event, неудавшуюся compaction. Но **надёжно отличить синтетические нули и неполный provider usage от полного нормализованного объекта не позволяет**. Простая проверка суммы или `totalTokens > 0` недостаточна.

Варианты для будущей реализации, без выбора нового механизма здесь: сохранять минимальную валидацию/provenance usage на provider boundary либо сверять generation stats по `responseId`. В адаптере есть `onProviderStreamEvent` до нормализации ([код](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/ai/src/api/openai-completions.ts#L544-L565)), но его доставка через текущий RPC bridge и покрытие внутренних summaries не проверены; это не готовая гарантия. Полный payload/текст для accounting публиковать не нужно.

При неизвестном usage следовать уже принятому контракту: запрет новых запросов, инфраструктурная причина остановки исследования, фиксация доступного `solution.py`, независимое воспроизведение, сохранение подтверждённых достижений; соблюдение токенного бюджета unknown. Не объявлять всё решение false только из-за accounting и не запускать исследование заново. Запоздалая сверка стоимости не должна разрешать продолжение исследования задним числом.

## Стоимость отдельно от токенного бюджета

[Тип Usage](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/ai/src/types.ts#L467-L495) содержит `cost.totalFromProvider?: boolean`: true означает provider-reported total, false/absence — каталоговую оценку. Краткая локальная документация message-types этот флаг не показывает, но исходники версии его содержат.

- [calculateCost](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/ai/src/models.ts#L1279-L1300) считает каталоговую оценку по тарифам за миллион токенов, отдельно input/output/cache; может выбирать ценовой tier по длине входа. Для обычного split: `USD_est = (I*pI + O*pO + R*pR + W*pW)/1e6`. Тарифы, модель, источник/время каталога нужно сохранять; это не независимое подтверждение списания.
- [OpenRouter override](https://github.com/earendil-works/pi/blob/b4afdddf44e059c67ccf5097e5317d913521b470/packages/ai/src/utils/openrouter-cost.ts) заменяет только `cost.total`: при валидном `cost` и `is_byok === false` — списанием OpenRouter; при BYOK — `cost + upstream_inference_cost` (или суммой prompt/completion upstream cost, если общий upstream равен null). При неполных данных оставляет estimate. Признак true не означает, что все компоненты `cost.*` теперь фактические.
- Поэтому нельзя заново складывать `cost.input + output + cacheRead + cacheWrite` вместо provider total или прибавлять total к компонентам. Поля компонентов остаются каталоговыми и могут не совпасть с фактическим total. Для BYOK полезно различать списание OpenRouter и суммарную стоимость OpenRouter + upstream: одно поле pi не хранит оба отдельно.
- [FAQ OpenRouter](https://openrouter.ai/docs/faq) определяет credits как баланс с базовой валютой USD; [OpenAPI](https://openrouter.ai/openapi.json) прямо описывает generation `total_cost` и `usage` в USD. Покупка credits и её комиссия — отдельные операции, не дополнительные токены/стоимость каждого inference. Публичная [документация моделей](https://openrouter.ai/docs/guides/overview/models) задаёт pricing в USD за token/request/unit, причём каталог отражает top provider, а не единый тариф всех upstream. Поэтому фиксированная цена по model ID не равна фактическому счёту.
- Unknown cost не 0. Подтверждённый нулевой тариф/нулевое списание допустимы, но нули по умолчанию ничего не доказывают. Успешный токенный accounting при отсутствии цены не требует инфраструктурной остановки: денежный лимит не принят.

Публичный unauthenticated `GET https://openrouter.ai/api/v1/models` ответил HTTP 200 с `Date: Sat, 14 Nov 2026 16:23:56 GMT`; найден точный ID `deepseek/deepseek-v4.1-flash-1017`, context length 1 048 576. Снимок pricing: `prompt="0.000000038"`, `completion="0.0000005"`, `input_cache_read="0.00000001"` за токен (соответственно $0.038/$0.50/$0.01 за миллион); cache-write цена отсутствует. Это публичный изменяемый каталог, **не** подтверждение реальной цены аккаунта, upstream, наличия модели в изолированном offline-каталоге pipeline или фактического списания. Не достраивать отсутствующую cache-write цену предположением.

## Предлагаемый текст resolution для родительской сессии

> Публичный контракт OpenRouter и pi 1.5.2 содержит данные для мягкого бюджета. Для исходного OpenRouter считать prompt_tokens + completion_tokens; для нормализованного pi — input + cacheRead + cacheWrite + output. Reasoning и cacheWrite1h не добавлять повторно. Каждый ответ и summary учитывать ровно один раз, включая доступный расход неуспешных обращений; накопительные streaming snapshots и session totals не складывать между собой. Контроль после каждого модельного ответа, включая внутренние summaries; при сумме >= 200 000 следующий модельный запрос не разрешать. Это целевой контракт, а не уже реализованная гарантия.
>
> Наличие pi Usage не доказывает полноту: адаптер подставляет нули при отсутствии provider fields. Для реализации необходимо подтверждение обязательных счётчиков на provider boundary либо достоверная независимая сверка; без него полнота и соблюдение бюджета unknown. При отсутствии/неполноте — инфраструктурная остановка исследования, фиксация доступной сдачи и её проверка по ранее согласованному контракту. Текущий RPC-клиент такой проверки и запрета внутренних запросов не обеспечивает; точку интеграции потребуется проверить при реализации.
>
> Стоимость хранить отдельно: provider-reported, estimated или unknown; использовать cost.totalFromProvider, учитывать специфику BYOK и не складывать provider total с каталоговыми компонентами. Неизвестную стоимость не считать нулём; денежного лимита нет. Конкретные usage/стоимость текущего аккаунта и корректность runtime не подтверждены — inference не запускался.

## Проверки и ограничения

Сетевые действия — публичный GitHub source/tag и публичный каталог/документация OpenRouter без credentials. Платные inference-запросы, Docker, живой pipeline, приватные журналы/сессии и account-specific generation endpoint не запускались/не читались. Клон исходников pi не завершился по таймауту; нужные файлы прочитаны отдельно через GitHub API по фиксированному commit. Установка не изменялась.

Перед реализационной приёмкой нужны синтетические случаи: cache read+write без двойного учёта; reasoning subset; несколько terminal представлений одного ответа; cumulative streaming; missing обязательное поле при положительном total; разрыв до usage; invalid split; compaction из двух запросов и достижение лимита после первого; BYOK/не-BYOK/estimate и валидный нулевой cost. Затем отдельный явно разрешённый smoke-test фактического профиля: terminal usage, модель/responseId, compaction, граница запрета следующего запроса. Исследование подтверждает публичный контракт, но не заменяет эти проверки.
