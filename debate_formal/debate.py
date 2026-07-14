"""
debate.py - CLI: punto de entrada para correr un debate formal completo entre
el Fiscal y el Defensor sobre una afirmacion de aritmetica / teoria de
numeros, arbitrado objetivamente por Z3 (ver orchestrator.py).

Uso:
    python debate.py "Para todo entero n, n al cubo menos n es divisible por 6."
    python debate.py --max-rondas 2 "n al cuadrado mas n mas 41 es primo para todo n >= 0."
    python debate.py
        (si no se pasa la afirmacion por argumento, la pide interactivamente)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from orchestrator import MAX_RONDAS, run_debate

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # En consolas Windows (cp1252) los LLMs a veces generan simbolos (∀, ∃,
    # etc.) que no se pueden imprimir con el codec por defecto. Forzamos UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="debate.py",
        description=(
            "Debate estructurado entre agentes de IA (Fiscal vs Defensor) sobre una "
            "afirmacion de aritmetica/teoria de numeros, arbitrado objetivamente por Z3 "
            "(nunca por un LLM decidiendo 'quien convencio mas')."
        ),
    )
    parser.add_argument(
        "afirmacion",
        nargs="?",
        default=None,
        help="Afirmacion a debatir, en lenguaje natural (si se omite, se pide por input).",
    )
    parser.add_argument(
        "--max-rondas",
        type=int,
        default=MAX_RONDAS,
        help=f"Maximo de rondas de debate Fiscal/Defensor (default: {MAX_RONDAS}).",
    )
    args = parser.parse_args()

    afirmacion = args.afirmacion
    if not afirmacion:
        try:
            afirmacion = input("Afirmacion a debatir: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nCancelado.")
            sys.exit(1)

    if not afirmacion:
        print("Error: no se proporciono ninguna afirmacion.", file=sys.stderr)
        sys.exit(1)

    if args.max_rondas < 1:
        print("Error: --max-rondas debe ser al menos 1.", file=sys.stderr)
        sys.exit(1)

    run_debate(afirmacion, max_rondas=args.max_rondas)


if __name__ == "__main__":
    main()
