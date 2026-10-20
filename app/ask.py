"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする

ツールも会話履歴も持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter

import httpx
from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


@dataclass
class Measurement:
    """1回のAPI呼び出しの結果。取得できない数値は0と区別してNoneにする。"""

    model: str
    question: str
    started_at: str
    elapsed_seconds: float
    status: str
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    finish_reason: str | None = None
    error_code: int | None = None
    error_type: str | None = None


def create_client() -> genai.Client:
    # SDK内部の再試行で429や待ち時間を隠さない。接続待ちも有限にする。
    return genai.Client(
        api_key=get_api_key(),
        http_options=types.HttpOptions(
            timeout=60_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def measure(
    client: genai.Client,
    question: str,
    *,
    model: str = MODEL,
    config: types.GenerateContentConfig | None = None,
) -> Measurement:
    """API呼び出しから全応答の受信までを計測する（初回トークンの時間ではない）。"""
    started_at = datetime.now(UTC).isoformat()
    start = perf_counter()
    try:
        response = client.models.generate_content(
            model=model, contents=question, config=config
        )
    except (errors.APIError, httpx.HTTPError) as exc:
        elapsed = perf_counter() - start
        # 例外本文にはURL等が含まれることがあるのでログに保存しない。
        return Measurement(
            model, question, started_at, elapsed, "error",
            error_code=exc.code if isinstance(exc, errors.APIError) else None,
            error_type=type(exc).__name__,
        )
    elapsed = perf_counter() - start
    usage = response.usage_metadata
    text = response.text or ""
    reason = response.candidates[0].finish_reason if response.candidates else None
    return Measurement(
        model, question, started_at, elapsed, "ok" if text.strip() else "empty",
        text=text,
        input_tokens=usage.prompt_token_count if usage else None,
        output_tokens=usage.candidates_token_count if usage else None,
        thinking_tokens=usage.thoughts_token_count if usage else None,
        finish_reason=reason.value if reason else None,
    )


def ask(question: str) -> str:
    with create_client() as client:
        result = measure(client, question)
    print(f"[応答時間] {result.elapsed_seconds:.3f} 秒")
    if result.status == "error":
        if result.error_code is not None:
            raise SystemExit(api_error_message(result.error_code))
        raise SystemExit(f"通信エラー: {result.error_type}")
    print(
        f"[トークン数] 入力: {result.input_tokens if result.input_tokens is not None else '取得不可'}"
        f" / 出力: {result.output_tokens if result.output_tokens is not None else '取得不可'}"
        f" / 思考: {result.thinking_tokens if result.thinking_tokens is not None else '取得不可'}"
    )
    if result.status == "empty":
        raise SystemExit("回答本文を取得できませんでした。")
    return result.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    print(ask(args.question))


if __name__ == "__main__":
    main()
