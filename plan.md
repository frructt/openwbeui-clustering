# plan.md

## 1. Контекст

Есть выгрузка диалогов пользователей из корпоративного ИИ-хаба на базе OpenWebUI.

Схема входных данных:

| Время | Пользователь | Chat UUID | Чат | role | message |

Объем данных: около 24 993 строк.

Особенности данных:
- все пользователи русскоязычные;
- `Чат` — это автосгенерированный заголовок OpenWebUI, использовать его только как вспомогательный сигнал;
- диалоги multi-turn, внутри одного `Chat UUID` может быть несколько разных задач;
- на первом этапе нужен быстрый практический результат, без ручной разметки;
- цель — понять, о чем общаются пользователи, какие темы самые массовые, какие сценарии растут, где есть признаки боли и какие из этого следуют бизнесовые инсайды.

Локально доступны модели:
- `Qwen/Qwen3-Embedding-8B` — для эмбеддингов;
- `gpt-oss-120b` — для постобработки, именования тем, кратких выводов и генерации бизнесовых summary.

Нужно реализовать оффлайн-пайплайн анализа логов, который:
1. загружает таблицу;
2. подготавливает данные;
3. строит embedding-based тематический анализ;
4. формирует карту тем и сводные отчеты;
5. извлекает первичные бизнесовые инсайды;
6. сохраняет результаты в удобном виде для последующей проверки.

---

## 2. Цель проекта

Сделать MVP-пайплайн тематического анализа пользовательских диалогов без ручной разметки.

Итог должен отвечать на вопросы:

1. Какие темы общения в ИИ-хабе самые частые?
2. Какие темы охватывают больше всего уникальных пользователей?
3. Какие темы растут по времени?
4. Какие темы связаны с кодингом, аналитикой, SQL, документами, Excel, Jira, Confluence, Grafana, LiteLLM, OpenWebUI и т.д.?
5. В каких темах есть признаки боли:
   - ошибки,
   - непонимание,
   - отсутствие функции,
   - проблемы с доступом,
   - неудобство UX,
   - длинные и тяжелые диалоги?
6. Какие из найденных тем тянут продуктовые решения:
   - новый режим,
   - новую интеграцию,
   - улучшение UX,
   - better onboarding,
   - better prompting,
   - улучшение качества модели?

---

## 3. Ограничения и принципы

### Обязательные ограничения
- не использовать ручную разметку в v1;
- не делать сложную кастомную ML-модель;
- не пытаться сразу строить идеальную taxonomy;
- не использовать `Чат` как основной объект анализа;
- не смешивать `user` и `assistant` в один корпус для кластеризации;
- не использовать `gpt-oss-120b` для классификации каждой строки по отдельности.

### Принципы реализации
- быстрый end-to-end результат важнее академической чистоты;
- пайплайн должен быть воспроизводимым;
- все промежуточные результаты должны сохраняться на диск;
- основная логика должна быть на Python;
- все параметры должны быть вынесены в конфиг;
- архитектура должна поддерживать дальнейшее развитие:
  - сегментацию диалогов,
  - ручную разметку,
  - LLM-based labeling,
  - dashboard/reporting.

---

## 4. Что считается успехом для v1

Система считается успешно реализованной, если она:

1. загружает исходную таблицу;
2. отбирает сообщения `role = user`;
3. строит embeddings через локальный `Qwen3-Embedding-8B`;
4. кластеризует тексты и выделяет темы;
5. формирует:
   - список тем,
   - ключевые слова,
   - representative examples,
   - размеры тем,
   - охват пользователей,
   - динамику по неделям;
6. дополнительно помечает темы простыми эвристиками по доменам:
   - code,
   - SQL,
   - Excel,
   - Jira,
   - Confluence,
   - Grafana,
   - LiteLLM,
   - OpenWebUI,
   - 1C,
   - документы/письма/презентации;
7. использует `gpt-oss-120b` только для:
   - генерации понятных названий тем,
   - краткого summary по теме,
   - гипотезы о JTBD и боли по representative examples;
8. сохраняет итоговый отчет в Markdown/CSV/Parquet.

---

## 5. Объект анализа

### V1: базовый режим
Объект анализа = одно сообщение пользователя (`role = user`).

Это самый быстрый старт.

### V1.1: улучшенный режим
Поддержать опциональный режим:
объект анализа = склейка 1–3 соседних user-сообщений в пределах одного `Chat UUID`, если:
- между ними разрыв не больше заданного порога;
- нет сильного сдвига темы;
- сообщений немного и они очевидно являются продолжением одного вопроса.

### Что сделать сейчас
Реализовать оба режима, но по умолчанию запускать V1 (одно user-сообщение).
Это даст быстрый и предсказуемый результат.

---

## 6. Общая архитектура пайплайна

Пайплайн должен состоять из следующих этапов:

1. **ingest**
   - загрузка CSV/Parquet/XLSX;
   - нормализация колонок;
   - базовая валидация.

2. **preprocess**
   - фильтрация `role = user`;
   - чистка текстов;
   - удаление пустых и почти пустых сообщений;
   - добавление технических признаков.

3. **unit_building**
   - формирование единиц анализа;
   - режим `message`;
   - режим `merged_messages`.

4. **embedding**
   - получение embeddings через локальный `Qwen3-Embedding-8B`;
   - батчинг;
   - ретраи;
   - кеширование результатов.

5. **topic_modeling**
   - UMAP;
   - HDBSCAN;
   - BERTopic поверх готовых embeddings;
   - выделение тем и outliers.

6. **topic_postprocessing**
   - ключевые слова;
   - representative examples;
   - доменные флаги;
   - статистики по теме.

7. **llm_enrichment**
   - использование `gpt-oss-120b` для:
     - topic title,
     - краткого описания темы,
     - гипотезы о JTBD,
     - гипотезы о боли,
     - бизнесового комментария.

8. **reporting**
   - summary tables;
   - trend tables;
   - examples tables;
   - итоговый Markdown-отчет.

---

## 7. Технический стек

Использовать Python 3.11+.

Основные библиотеки:
- `pandas`
- `numpy`
- `pyarrow`
- `scikit-learn`
- `umap-learn`
- `hdbscan`
- `bertopic`
- `openai` или совместимый клиент для OpenAI-compatible API
- `matplotlib` или `plotly` для простых графиков
- `typer` или `argparse` для CLI
- `pyyaml` для конфигов
- `tqdm` для прогресса

---

## 8. Предполагаемая структура проекта

```text
project/
  README.md
  plan.md
  pyproject.toml
  requirements.txt

  configs/
    default.yaml

  data/
    raw/
    interim/
    processed/

  reports/
    figures/
    tables/
    insights/

  notebooks/
    01_eda.ipynb
    02_topic_modeling.ipynb
    03_insights.ipynb

  src/
    __init__.py
    config.py
    io_utils.py
    preprocess.py
    unit_builder.py
    embedding_client.py
    topic_model.py
    domain_flags.py
    llm_enrichment.py
    metrics.py
    reporting.py
    pipeline.py
    cli.py

  tests/
    test_preprocess.py
    test_unit_builder.py
    test_domain_flags.py
    test_pipeline_smoke.py
```

---

## 9. Конфиг

Нужен YAML-конфиг с параметрами запуска.

Пример структуры:

```yaml
input:
  path: "data/raw/dialogs.csv"
  format: "csv"
  time_column: "Время"
  user_column: "Пользователь"
  chat_uuid_column: "Chat UUID"
  chat_title_column: "Чат"
  role_column: "role"
  message_column: "message"

processing:
  role_keep: "user"
  min_chars: 10
  drop_trivial_messages: true
  trivial_messages:
    - "ок"
    - "да"
    - "нет"
    - "спасибо"
    - "понял"

units:
  mode: "message"   # or "merged_messages"
  max_messages_per_unit: 3
  max_gap_minutes: 20

embeddings:
  base_url: "http://localhost:8000/v1"
  api_key: "dummy"
  model: "Qwen/Qwen3-Embedding-8B"
  batch_size: 64
  timeout_sec: 120
  max_retries: 5
  save_path: "data/interim/embeddings.npy"

topic_model:
  umap_n_neighbors: 15
  umap_n_components: 10
  umap_min_dist: 0.0
  umap_metric: "cosine"

  hdbscan_min_cluster_size: 30
  hdbscan_min_samples: 10
  hdbscan_cluster_selection_method: "eom"

  vectorizer_ngram_min: 1
  vectorizer_ngram_max: 2
  vectorizer_min_df: 5
  nr_topics: null

llm:
  enabled: true
  base_url: "http://localhost:8001/v1"
  api_key: "dummy"
  model: "gpt-oss-120b"
  max_examples_per_topic: 12

reporting:
  top_n_topics: 30
  top_n_examples: 15
  output_dir: "reports/insights"
```

---

## 10. Подробные требования по этапам

## 10.1. Ingest

### Нужно реализовать
- загрузку из CSV;
- загрузку из Parquet;
- опционально из XLSX;
- нормализацию названий колонок во внутреннюю схему:
  - `timestamp`
  - `user`
  - `chat_uuid`
  - `chat_title`
  - `role`
  - `message`

### Проверки
- обязательные колонки присутствуют;
- `timestamp` парсится;
- `message` не пустой;
- `chat_uuid` не пустой;
- сохранять краткий отчет о качестве данных:
  - сколько строк всего;
  - сколько `user`;
  - сколько `assistant`;
  - сколько пустых сообщений;
  - сколько уникальных пользователей;
  - сколько уникальных чатов.

---

## 10.2. Preprocess

### Нужно сделать
- оставить только `role = user`;
- удалить пустые сообщения;
- удалить сообщения короче заданного порога;
- нормализовать пробелы, переносы строк;
- убрать дубликаты, если они полностью совпадают по:
  - `chat_uuid`
  - `timestamp`
  - `user`
  - `message`

### Добавить технические признаки
- `message_len_chars`
- `message_len_words`
- `date`
- `week`
- `month`

### Простые бинарные признаки
Реализовать эвристики/regex-флаги:
- `has_code`
- `has_sql`
- `has_link`
- `has_excel_terms`
- `has_jira_terms`
- `has_confluence_terms`
- `has_grafana_terms`
- `has_litellm_terms`
- `has_openwebui_terms`
- `has_1c_terms`
- `has_error_terms`

Словари держать в отдельном модуле и сделать расширяемыми.

---

## 10.3. Формирование единиц анализа

### Режим `message`
Одна строка = один объект анализа.

### Режим `merged_messages`
Склеивать соседние user-сообщения в пределах одного `chat_uuid`, если:
- время между ними не превышает порог;
- число сообщений в одном объекте не превышает `max_messages_per_unit`.

### Для каждой единицы анализа сохранить
- `unit_id`
- `chat_uuid`
- `user`
- `chat_title`
- `start_ts`
- `end_ts`
- `n_messages`
- `text`
- агрегированные доменные флаги
- длину текста

---

## 10.4. Embeddings

### Обязательное требование
Использовать локальный `Qwen/Qwen3-Embedding-8B` через OpenAI-compatible endpoint.

### Реализовать
- клиент для embeddings;
- батчинг;
- ретраи;
- таймауты;
- логирование ошибок;
- сохранение результатов на диск;
- возможность повторно использовать уже посчитанные embeddings без повторного вызова модели.

### Требования к реализации
- вход: список текстов;
- выход: `numpy.ndarray` размерности `[N, D]`;
- порядок embeddings должен строго соответствовать порядку объектов анализа.

---

## 10.5. Topic modeling

### Основной подход
Использовать embedding-based тематическое моделирование:
- embeddings берутся из локального `Qwen3-Embedding-8B`;
- затем UMAP;
- затем HDBSCAN;
- затем BERTopic для интерпретируемых тем.

### Что нужно сделать
- обучить модель тем на всем корпусе;
- получить `topic_id` для каждого объекта анализа;
- получить outliers;
- получить ключевые слова по темам;
- получить representative examples.

### Стартовые параметры
Использовать стартовые параметры из конфига.

### Важно
Сделать код так, чтобы было легко прогнать 2–3 конфигурации:
- более крупные темы;
- более мелкие темы;
- сравнить долю outliers и интерпретируемость.

### Что сохранить
- сериализованную topic model;
- таблицу `unit_id -> topic_id`;
- таблицу summary по темам;
- таблицу examples по темам.

---

## 10.6. Постобработка тем

Для каждой темы нужно посчитать:

- `topic_id`
- число объектов анализа
- число уникальных пользователей
- число уникальных чатов
- среднюю длину текста
- долю текстов с признаками:
  - code
  - SQL
  - Excel
  - Jira
  - Confluence
  - Grafana
  - LiteLLM
  - OpenWebUI
  - 1C
  - error/problem

### Representative examples
Для каждой темы выбрать:
- 10–15 representative examples;
- по возможности самые типичные, а не случайные.

### Top keywords
Сохранить топ ключевых слов/фраз по теме.

---

## 10.7. LLM enrichment через `gpt-oss-120b`

### Важно
`gpt-oss-120b` использовать не на все 25k строк, а только на уровень тем.

### Для каждой темы передать в LLM
- `topic_id`
- топ ключевых слов
- 8–12 representative examples
- базовые статистики темы

### Что LLM должен вернуть
В JSON-формате:
- `topic_title`
- `topic_description`
- `suspected_jtbd`
- `suspected_pain_points`
- `suspected_business_value`
- `suspected_product_action`
- `confidence_note`

### Правила
- никаких выдуманных фактов;
- только гипотезы на основе representative examples;
- короткие и деловые формулировки;
- не делать из LLM окончательного judge;
- все поля явно помечать как `suspected`/гипотезы.

---

## 10.8. Reporting

Нужно сформировать следующие артефакты.

### Таблица 1. `topic_summary.csv`
Поля:
- `topic_id`
- `topic_title_auto`
- `topic_title_llm`
- `topic_keywords`
- `n_units`
- `n_users`
- `n_chats`
- `share_of_total`
- `avg_len_chars`
- `has_code_share`
- `has_sql_share`
- `has_excel_share`
- `has_jira_share`
- `has_confluence_share`
- `has_grafana_share`
- `has_litellm_share`
- `has_openwebui_share`
- `has_1c_share`
- `has_error_terms_share`

### Таблица 2. `topic_examples.csv`
Поля:
- `topic_id`
- `chat_uuid`
- `user`
- `timestamp`
- `text`

### Таблица 3. `topic_trends_by_week.csv`
Поля:
- `week`
- `topic_id`
- `n_units`

### Таблица 4. `topic_domain_breakdown.csv`
Поля:
- `topic_id`
- все доли доменных флагов

### Отчет `insight_report.md`
Должен содержать:
1. краткое описание датасета;
2. сколько сообщений вошло в анализ;
3. сколько тем выделено;
4. топ тем по объему;
5. топ тем по охвату пользователей;
6. топ растущих тем;
7. темы с наибольшей долей error/problem сигналов;
8. темы, связанные с:
   - code,
   - SQL,
   - Excel,
   - Jira/Confluence,
   - Grafana,
   - LiteLLM/OpenWebUI;
9. краткие бизнесовые гипотезы;
10. product implications.

---

## 11. Что считать “быстрыми бизнесовыми инсайтами”

Без ручной разметки pipeline должен как минимум извлекать такие сигналы:

1. **Топ сценарии использования**
   - кодинг,
   - аналитика,
   - SQL,
   - работа с документами,
   - объяснение понятий,
   - корпоративные интеграции.

2. **Признаки неудовлетворенного спроса**
   - длинные и сложные диалоги;
   - высокая доля error/problem-слов;
   - повторяющиеся темы с большим числом пользователей.

3. **Сигналы к roadmap**
   - нужен отдельный режим под SQL/Excel;
   - нужна интеграция с Jira/Confluence;
   - нужен better onboarding;
   - нужна доработка UX;
   - нужны better prompts/system prompts;
   - нужна better model routing/eval.

4. **Сигналы к операционным решениям**
   - какие темы массовые;
   - какие темы растут;
   - где люди тратят больше всего времени и итераций.

---

## 12. CLI

Сделать CLI-команду вида:

```bash
python -m src.cli run --config configs/default.yaml
```

Поддержать отдельные стадии:

```bash
python -m src.cli ingest --config configs/default.yaml
python -m src.cli preprocess --config configs/default.yaml
python -m src.cli embed --config configs/default.yaml
python -m src.cli topics --config configs/default.yaml
python -m src.cli enrich --config configs/default.yaml
python -m src.cli report --config configs/default.yaml
```

---

## 13. Минимальные тесты

Нужно сделать хотя бы smoke/unit tests на:

1. загрузку и нормализацию колонок;
2. фильтрацию `role = user`;
3. построение unit в режиме `message`;
4. построение unit в режиме `merged_messages`;
5. доменные флаги;
6. корректное соответствие embeddings количеству объектов анализа;
7. smoke-test полного пайплайна на маленьком sample dataset.

---

## 14. Что не нужно делать в этом спринте

Не делать сейчас:
- ручную разметку;
- полноценную иерархическую taxonomy;
- сложный dashboard;
- production API;
- real-time обработку;
- agentic pipeline;
- классификацию каждого сообщения через `gpt-oss-120b`;
- анализ `assistant`-ответов как отдельный тематический корпус.

---

## 15. Артефакты, которые должны появиться после реализации

Ожидаемые выходные файлы:

```text
data/interim/raw_normalized.parquet
data/interim/user_messages.parquet
data/interim/analysis_units.parquet
data/interim/embeddings.npy
data/interim/topic_assignments.parquet

reports/tables/topic_summary.csv
reports/tables/topic_examples.csv
reports/tables/topic_trends_by_week.csv
reports/tables/topic_domain_breakdown.csv

reports/insights/insight_report.md
reports/figures/topic_sizes.png
reports/figures/topic_growth.png
```

---

## 16. Порядок выполнения для Codex

1. Создать структуру проекта.
2. Реализовать ingest и preprocess.
3. Реализовать unit builder.
4. Реализовать embedding client под локальный `Qwen3-Embedding-8B`.
5. Реализовать topic modeling на BERTopic с precomputed embeddings.
6. Реализовать доменные флаги и метрики.
7. Реализовать LLM enrichment через `gpt-oss-120b`.
8. Реализовать reporting.
9. Добавить CLI.
10. Добавить минимальные тесты.
11. Проверить весь pipeline на sample.
12. Подготовить README с инструкцией запуска.

---

## 17. Definition of done

Задача считается выполненной, если:

- весь pipeline запускается локально одной командой;
- на выходе есть таблицы и Markdown-отчет;
- выделены интерпретируемые темы;
- у каждой крупной темы есть:
  - название,
  - ключевые слова,
  - representative examples,
  - размер,
  - охват пользователей,
  - базовые доменные флаги,
  - краткое бизнесовое summary;
- результаты можно использовать для первичного product/business review без ручной разметки.

---

## 18. Дополнительные пожелания по качеству кода

- писать чистый и понятный Python;
- не дублировать логику;
- использовать dataclass / typed config where useful;
- важные функции покрыть type hints;
- писать логирование по этапам;
- не держать магические числа внутри кода;
- все параметры вынести в конфиг;
- не падать молча: все ошибки должны логироваться понятно.

---

## 19. README: что описать

В README нужно кратко описать:
- назначение проекта;
- формат входных данных;
- как настроить локальные endpoint для:
  - `Qwen/Qwen3-Embedding-8B`
  - `gpt-oss-120b`
- как запустить pipeline;
- какие файлы будут на выходе;
- как интерпретировать результаты.
