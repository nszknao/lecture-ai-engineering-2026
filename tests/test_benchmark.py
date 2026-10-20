"""第1回 発展: 計測・ログ・集計をAPIへの接続なしで検証する。"""

import argparse
import json
from dataclasses import asdict
from unittest.mock import MagicMock

import httpx
import pytest
from google.genai import errors, types

from app import ask
from scripts import benchmark


def response(usage=None, text="回答", reason="STOP"):
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(parts=[types.Part(text=text)]),
                finish_reason=reason,
            )
        ],
        usage_metadata=usage,
    )


@pytest.mark.parametrize(
    "usage, expected",
    [
        ({"prompt_token_count": 12, "candidates_token_count": 34, "thoughts_token_count": 5},
         (12, 34, 5)),
        ({"prompt_token_count": 12, "candidates_token_count": 0}, (12, 0, None)),
        (None, (None, None, None)),
    ],
)
def test_measure_records_wall_time_and_missing_usage(monkeypatch, usage, expected):
    client = MagicMock()
    client.models.generate_content.return_value = response(usage)
    ticks = iter([10.0, 11.25])
    monkeypatch.setattr(ask, "perf_counter", lambda: next(ticks))
    result = ask.measure(client, "質問", model="test-model")
    assert result.elapsed_seconds == 1.25
    assert (result.input_tokens, result.output_tokens, result.thinking_tokens) == expected
    assert result.status == "ok"
    assert result.text == "回答"
    assert result.finish_reason == "STOP"
    assert result.started_at.endswith("+00:00")
    client.models.generate_content.assert_called_once_with(
        model="test-model", contents="質問", config=None
    )


@pytest.mark.parametrize(
    "error, code",
    [(errors.ClientError(429, {"message": "secret-url"}), 429),
     (httpx.ReadTimeout("secret-url"), None)],
)
def test_measure_records_failures_without_exception_text(error, code):
    client = MagicMock()
    client.models.generate_content.side_effect = error
    result = ask.measure(client, "質問")
    assert result.status == "error"
    assert result.error_code == code
    assert result.elapsed_seconds >= 0
    assert "secret-url" not in json.dumps(asdict(result))
    client.models.generate_content.assert_called_once()


def test_measure_handles_empty_response():
    client = MagicMock()
    client.models.generate_content.return_value = types.GenerateContentResponse()
    result = ask.measure(client, "質問")
    assert result.status == "empty"
    assert result.text == ""
    assert result.finish_reason is None


def test_ask_keeps_text_return_and_displays_metrics(monkeypatch, capsys):
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.return_value = response(
        {"prompt_token_count": 12, "candidates_token_count": 0}
    )
    monkeypatch.setattr(ask, "create_client", lambda: client)
    assert ask.ask("質問") == "回答"
    output = capsys.readouterr().out
    assert "[応答時間]" in output
    assert "入力: 12 / 出力: 0 / 思考: 取得不可" in output
    client.__exit__.assert_called_once()


def test_client_disables_retries_and_has_timeout(monkeypatch):
    factory = MagicMock()
    monkeypatch.setattr(ask.genai, "Client", factory)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    ask.create_client()
    options = factory.call_args.kwargs["http_options"]
    assert options.retry_options.attempts == 1
    assert options.timeout == 60_000


def arguments(tmp_path, **overrides):
    values = dict(
        models=["model-a", "model-b"], questions=None, repeats=3,
        interval=30.0, temperature=1.0, max_output_tokens=2048,
        output=tmp_path / "results.jsonl",
    )
    return argparse.Namespace(**(values | overrides))


def setup_run(monkeypatch, side_effect):
    client = MagicMock()
    client.__enter__.return_value = client
    client.models.generate_content.side_effect = side_effect
    monkeypatch.setattr(benchmark, "create_client", lambda: client)
    sleep = MagicMock()
    monkeypatch.setattr(benchmark.time, "sleep", sleep)
    return client, sleep


def test_run_saves_all_trials_and_alternates_order(monkeypatch, tmp_path):
    client, sleep = setup_run(monkeypatch, [response()] * 18)
    args = arguments(tmp_path)
    assert benchmark.run(args) == 0
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == 18
    assert [row["model"] for row in rows[6:8]] == ["model-b", "model-a"]
    assert {row["repeat"] for row in rows} == {1, 2, 3}
    assert all(row["config"] == {"temperature": 1.0, "max_output_tokens": 2048}
               for row in rows)
    assert all(row["question"] == benchmark.DEFAULT_QUESTIONS[row["question_id"]]
               for row in rows)
    assert client.models.generate_content.call_count == 18
    assert sleep.call_count == 17
    sleep.assert_called_with(30.0)


def test_rate_limit_is_saved_and_stops_run(monkeypatch, tmp_path):
    client, sleep = setup_run(
        monkeypatch, [response(), errors.ClientError(429, {"message": "quota"})]
    )
    args = arguments(tmp_path)
    assert benchmark.run(args) == 1
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[1]["status"] == "error"
    assert rows[1]["error_code"] == 429
    assert client.models.generate_content.call_count == 2
    sleep.assert_called_once()


def test_existing_log_is_overwritten(monkeypatch, tmp_path):
    client, _ = setup_run(monkeypatch, [response()] * 3)
    args = arguments(tmp_path, models=["model-a"], repeats=1)
    args.output.write_text("existing data\n")
    assert benchmark.run(args) == 0
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert len(rows) == 3
    assert all(row["status"] == "ok" for row in rows)
    assert client.models.generate_content.call_count == 3


def test_summary_excludes_errors_and_missing_values(tmp_path):
    base = dict(
        model="model-a", question_id="short", question="質問", config={},
        status="ok", elapsed_seconds=2.0, input_tokens=10, output_tokens=0,
        thinking_tokens=None, finish_reason="STOP",
    )
    rows = [
        base,
        base | dict(elapsed_seconds=4.0, input_tokens=None, output_tokens=20,
                    finish_reason="MAX_TOKENS"),
        base | dict(status="error", elapsed_seconds=100.0, input_tokens=999),
        base | dict(status="empty", elapsed_seconds=80.0, input_tokens=999),
    ]
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    report = benchmark.summarize(path)
    assert "| 2/4 | 1 | 3.000 | 2.000〜4.000 | 10.0 (n=1) | 10.0 (n=2) | — |" in report


def test_summary_keeps_different_settings_separate(monkeypatch, tmp_path):
    setup_run(monkeypatch, [response()] * 18)
    args = arguments(tmp_path)
    benchmark.run(args)
    rows = [json.loads(line) for line in args.output.read_text().splitlines()]
    rows[0]["config"]["temperature"] = 0.5
    args.output.write_text("\n".join(json.dumps(row) for row in rows))
    report = benchmark.summarize(args.output)
    assert len([line for line in report.splitlines() if line.startswith("| model-")]) == 7


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_intervals_are_rejected(value):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark.nonnegative_float(value)


def test_custom_questions_must_be_nonempty_strings(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text('{"question": 123}')
    with pytest.raises(ValueError):
        benchmark.load_questions(path)
