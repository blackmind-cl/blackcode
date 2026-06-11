"""`blackcode generate-qa` — genera pares pregunta/respuesta desde un corpus."""

from __future__ import annotations

import argparse


def _run(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from blackcode.generate import generate_qa

    corpus = Path(args.corpus)
    records = [
        json.loads(line)
        for line in corpus.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = generate_qa(
        records,
        model_id=args.model,
        device=args.device,
        per_chunk=args.per_chunk,
        chunk_size=args.chunk_size,
    )
    out = Path(args.output)
    with out.open("w", encoding="utf-8") as fh:
        for row in dataset:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Generados {len(dataset)} pares Q&A -> {out}")
    return 0


def register(sub) -> None:
    p = sub.add_parser("generate-qa", help="Generar pares Q&A desde un corpus")
    p.add_argument(
        "corpus",
        nargs="?",
        default="corpus.jsonl",
        help="JSONL de corpus (de 'blackcode ingest')",
    )
    p.add_argument("--output", default="dataset.jsonl", help="JSONL de salida")
    p.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Modelo generador local",
    )
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Dispositivo para la generación",
    )
    p.add_argument(
        "--per-chunk", type=int, default=3, help="Pares Q&A por trozo de texto"
    )
    p.add_argument(
        "--chunk-size", type=int, default=2000, help="Tamaño de trozo (caracteres)"
    )
    p.set_defaults(func=_run)
