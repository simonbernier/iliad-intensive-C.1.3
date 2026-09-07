import json
import random
import asyncio
from pathlib import Path

from lib.sampling import make_client, sample
from lib.parsing import parse_values
from lib.scoring import summarize, exact_match, value_accuracy


async def _run_one(client, model, main_task, values, max_tokens, temperature,
                   system_prompt, encoder_template, decoder_template, scheme):
    answer = await sample(
        client, model,
        encoder_template.format(main_task=main_task, values=" ".join(map(str, values)), scheme=scheme),
        system_prompt, max_tokens=max_tokens, temperature=temperature,
    )
    decoded_raw = await sample(
        client, model,
        decoder_template.format(answer=answer, scheme=scheme),
        max_tokens=max_tokens, temperature=temperature,
    )
    decoded = parse_values(decoded_raw)[: len(values)]
    return {"main_task": main_task, "values": values, "answer": answer,
            "decoded_raw": decoded_raw, "decoded": decoded}


def _write_completions(path, results):
    """One JSON row per sample: prompt, target values, answer, decode, per-sample scores."""
    with path.open("w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            row = {
                "index": i,
                "main_task": r["main_task"],
                "values": r["values"],
                "answer": r["answer"],
                "decoded_raw": r["decoded_raw"],
                "decoded": r["decoded"],
                "exact": exact_match(r["decoded"], r["values"]),
                "value_accuracy": value_accuracy(r["decoded"], r["values"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_metrics(path, metrics, results, model, n_values, value_range, temperature, seed):
    """Aggregate scores plus the run settings needed to interpret them later."""
    payload = {
        "model": model,
        "n_eval": len(results),
        "n_values": n_values,
        "value_range": value_range,
        "temperature": temperature,
        "seed": seed,
        "n_exact": sum(exact_match(r["decoded"], r["values"]) for r in results),
        **metrics,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


async def _run_steg(model, n_eval, n_values, value_range, max_tokens, temperature, seed,
                    system_prompt, encoder_template, decoder_template, scheme, main_tasks,
                    output_dir):
    rng = random.Random(seed)
    client = make_client()
    samples = [
        (rng.choice(main_tasks), [rng.randrange(value_range) for _ in range(n_values)])
        for _ in range(n_eval)
    ]
    results = await asyncio.gather(*[
        _run_one(client, model, mt, vals, max_tokens, temperature,
                 system_prompt, encoder_template, decoder_template, scheme)
        for mt, vals in samples
    ])

    metrics = summarize([(r["decoded"], r["values"]) for r in results])

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_completions(out_dir / "completions.jsonl", results)
    _write_metrics(out_dir / "metrics.json", metrics, results, model, n_values,
                   value_range, temperature, seed)

    for i, r in enumerate(results):
        print(f"sample {i + 1}/{len(results)}: values={r['values']} "
              f"decoded={r['decoded']} exact={exact_match(r['decoded'], r['values'])}")

    print("\n=== summary ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
    print(f"\nwrote {out_dir / 'completions.jsonl'} and {out_dir / 'metrics.json'}")


def run_steg(model, n_eval, n_values, value_range, max_tokens, temperature, seed,
             system_prompt, encoder_template, decoder_template, scheme, main_tasks,
             output_dir):
    asyncio.run(_run_steg(model, n_eval, n_values, value_range, max_tokens, temperature,
                          seed, system_prompt, encoder_template, decoder_template,
                          scheme, main_tasks, output_dir))
