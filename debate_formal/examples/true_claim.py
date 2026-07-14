"""
Ejemplo de afirmacion VERDADERA: "para todo entero n, n^3 - n es divisible por 6".

Se espera que el Defensor "gane" -- ni el chequeo base de la afirmacion ni
ningun ataque del Fiscal a lo largo de las rondas deberian producir un
contraejemplo real confirmado por Z3, asi que el veredicto final deberia ser
VERDADERA.

Correr con:
    python examples/true_claim.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from orchestrator import run_debate

if __name__ == "__main__":
    resultado = run_debate("Para todo entero n, n al cubo menos n es divisible por 6.", max_rondas=3)
    print(f"\n>>> Veredicto obtenido: {resultado.veredicto} (esperado: VERDADERA)")
