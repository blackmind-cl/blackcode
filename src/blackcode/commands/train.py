"""`blackcode train` / `blackcode eval` — pipeline de entrenamiento y evaluación."""

from __future__ import annotations

import argparse

from blackcode.commands import PROJECT_CONFIG


def _run_train(args: argparse.Namespace) -> int:
    import blackcode.trainers  # noqa: F401  (registra los de fábrica)
    from blackcode.config import load_config
    from blackcode.evaluation.metrics import summarize
    from blackcode.pipeline import Pipeline

    config = load_config(args.config)
    if args.resume:
        config.train.resume = True
    print(f"Ejecutando '{config.name}' (trainer={config.train.trainer})")
    result = Pipeline(config).run(do_eval=not args.no_eval)
    print("\nEntrenamiento completado. Métricas de entrenamiento:")
    print(summarize(result.train.metrics))
    if result.eval_metrics:
        print("\nMétricas de evaluación:")
        print(summarize(result.eval_metrics))
    print(f"\nModelo guardado en: {result.train.output_dir}")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    import blackcode.trainers  # noqa: F401
    from blackcode.config import load_config
    from blackcode.evaluation.metrics import summarize
    from blackcode.pipeline import Pipeline

    config = load_config(args.config)
    print(f"Evaluando '{config.name}' (modelo en {config.train.output_dir})")
    metrics = Pipeline(config).evaluate()
    print("\nMétricas de evaluación:")
    print(summarize(metrics))
    return 0


def register(sub) -> None:
    p_train = sub.add_parser("train", help="Ejecutar el pipeline")
    p_train.add_argument("config", nargs="?", default=PROJECT_CONFIG)
    p_train.add_argument("--no-eval", action="store_true", help="Omitir evaluación")
    p_train.add_argument(
        "--resume", action="store_true", help="Reanudar desde el último checkpoint"
    )
    p_train.set_defaults(func=_run_train)

    p_eval = sub.add_parser("eval", help="Evaluar un modelo ya entrenado")
    p_eval.add_argument("config", nargs="?", default=PROJECT_CONFIG)
    p_eval.set_defaults(func=_run_eval)
