"""第1回 発展: 同じ条件でモデルを比較し、結果をJSONLに保存する。

実行例（GEMINI_API_KEYは環境変数に設定）:
    uv run python -m scripts.benchmark run --models MODEL_A MODEL_B \
        --output lectures/01/results.jsonl
    uv run python -m scripts.benchmark summarize lectures/01/results.jsonl

質問を変える場合は --questions で {"質問ID": "質問本文", ...} のJSONを指定する。
同じ保存先の結果は実行ごとに上書きする。途中停止した結果も summarize で集計できる。
"""

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median

from google.genai import types

from app.ask import create_client, measure

DEFAULT_QUESTIONS = {
    "short": "LoRAとは何ですか？日本語で100字程度で説明してください。",
    "comparison": (
        "RAGとファインチューニングの違いを、目的・必要なデータ・更新方法の"
        "3点から比較し、使い分けの例も含めて日本語で600字程度で説明してください。"
    ),
    "course": (
        "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
        "日本語で200字以内で回答し、分からない場合は分からないと答えてください。"
    ),
}


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("1以上の整数を指定してください")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0以上の有限の数を指定してください")
    return number


def load_questions(path: Path | None) -> dict[str, str]:
    questions = DEFAULT_QUESTIONS if path is None else json.loads(path.read_text())
    if not isinstance(questions, dict) or not questions or not all(
        isinstance(key, str) and key.strip()
        and isinstance(value, str) and value.strip()
        for key, value in questions.items()
    ):
        raise ValueError("質問は空でないJSONオブジェクト {質問ID: 質問本文} にしてください")
    return questions


def run(args: argparse.Namespace) -> int:
    questions = load_questions(args.questions)
    models = list(dict.fromkeys(args.models))
    config = types.GenerateContentConfig(
        temperature=args.temperature, max_output_tokens=args.max_output_tokens
    )
    total = len(models) * len(questions) * args.repeats
    config_record = config.model_dump(mode="json", exclude_none=True)
    print(f"最大{total}回 / 呼び出し終了後の間隔{args.interval}秒 / 保存先: {args.output}")

    # クライアントの初期化後、APIを呼ぶ前に保存先を上書きで開く。
    with create_client() as client, args.output.open("w", encoding="utf-8") as stream:
        count = 0
        failed = False
        for repeat in range(1, args.repeats + 1):
            for question_id, question in questions.items():
                # 反復ごとに順序を反転し、常に同じモデルが先になる偏りを軽減する。
                order = models if repeat % 2 else list(reversed(models))
                for model in order:
                    if count:
                        time.sleep(args.interval)
                    result = measure(client, question, model=model, config=config)
                    record = {
                        "schema_version": 1,
                        "question_id": question_id,
                        "repeat": repeat,
                        "config": config_record,
                        "interval_seconds": args.interval,
                        "sdk_attempts": 1,
                        "timeout_seconds": 60,
                        **asdict(result),
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    count += 1
                    print(
                        f"[{count}/{total}] {model} / {question_id} / {repeat}回目: "
                        f"{result.status}, {result.elapsed_seconds:.3f}秒",
                        flush=True,
                    )
                    if result.status != "ok":
                        failed = True
                    if result.status == "error":
                        # 上限・設定・通信の問題を記録して停止する。自動再試行しない。
                        print(
                            f"エラーで停止: {result.error_type} "
                            f"(HTTP {result.error_code})。保存済みの結果を確認してください。"
                        )
                        return 1
    return int(failed)


def summarize(path: Path) -> str:
    groups = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            # 同名IDでも質問本文や設定が違う計測を混ぜない。
            key = (
                row["model"], row["question_id"], row["question"],
                json.dumps(row["config"], sort_keys=True, ensure_ascii=False),
            )
            groups[key].append(row)

    lines = [
        "| モデル | 質問ID | 設定 | 成功/試行 | 打切り | 中央値 秒 | 最小〜最大 秒 | 平均入力 | 平均出力 | 平均思考 |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]

    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    for (model, question_id, question, config), rows in groups.items():
        successful = [row for row in rows if row["status"] == "ok"]
        times = [row["elapsed_seconds"] for row in successful]

        def average(field: str) -> str:
            values = [row[field] for row in successful if row[field] is not None]
            return f"{mean(values):.1f} (n={len(values)})" if values else "—"

        mid = f"{median(times):.3f}" if times else "—"
        span = f"{min(times):.3f}〜{max(times):.3f}" if times else "—"
        truncated = sum(row["finish_reason"] == "MAX_TOKENS" for row in successful)
        lines.append(
            f"| {cell(model)} | {cell(question_id)}: {cell(question)} | {cell(config)} "
            f"| {len(successful)}/{len(rows)} | {truncated} | {mid} | {span} "
            f"| {average('input_tokens')} | {average('output_tokens')} "
            f"| {average('thinking_tokens')} |"
        )
    lines.extend([
        "",
        "時間とトークン数は本文のある成功応答だけを集計。欠測値は平均から除き、nは取得件数。",
        "打切りはMAX_TOKENSの件数（成功の集計にも含む）。回答品質は本文と終了理由を別途確認する。",
        "応答時間は通信を含む全応答の受信時間。モデル内部だけの処理速度や初回トークンの時間ではない。",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="APIを呼び出して計測する")
    runner.add_argument("--models", nargs="+", required=True)
    runner.add_argument("--questions", type=Path)
    runner.add_argument("--repeats", type=positive_int, default=3)
    runner.add_argument("--interval", type=nonnegative_float, default=30)
    runner.add_argument("--temperature", type=nonnegative_float, default=1.0)
    runner.add_argument("--max-output-tokens", type=positive_int, default=2048)
    runner.add_argument("--output", type=Path, required=True)
    summary = commands.add_parser("summarize", help="保存済みの結果をAPIなしで集計する")
    summary.add_argument("path", type=Path)
    args = parser.parse_args()
    if args.command == "run" and args.temperature > 2:
        parser.error("temperatureは0〜2で指定してください")
    try:
        if args.command == "run":
            raise SystemExit(run(args))
        print(summarize(args.path))
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
