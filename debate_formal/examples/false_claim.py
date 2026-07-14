"""
Ejemplo de afirmacion FALSA: "n^2 + n + 41 es primo para todo n >= 0".

Esta es la famosa formula de Euler que genera primos para n = 0..39, pero
falla en n = 40: 40^2 + 40 + 41 = 1681 = 41 x 41 (no es primo). Se espera que
el Fiscal (o el chequeo base de la afirmacion) encuentre este contraejemplo
real, confirmado por Z3, y el veredicto final deberia ser FALSA.

Correr con:
    python examples/false_claim.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orchestrator import run_debate

if __name__ == "__main__":
    resultado = run_debate(
        "n al cuadrado mas n mas 41 es un numero primo para todo n mayor o igual a 0.",
        max_rondas=3,
    )
    print(f"\n>>> Veredicto obtenido: {resultado.veredicto} (esperado: FALSA)")
    print(f">>> Contraejemplo: {resultado.contraejemplo} (esperado: n=40)")
