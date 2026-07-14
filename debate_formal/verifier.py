"""
verifier.py - Ejecuta codigo Z3 generado por el Formalizador en un subproceso
aislado con timeout, y devuelve un veredicto estructurado (sat/unsat/unknown).

Este modulo es el arbitro objetivo del sistema: NO usa LLMs, es codigo puro.
El codigo Z3 nunca se ejecuta con eval/exec en el proceso principal -- siempre
corre en un subproceso nuevo, con timeout, para evitar que un loop infinito
(o codigo malicioso) cuelgue el programa principal.
"""

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class VerdictStatus(str, Enum):
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class VerifierResult:
    status: VerdictStatus
    model: Optional[Dict[str, str]] = None
    error: Optional[str] = None
    raw_stdout: str = ""
    raw_stderr: str = ""

    def __repr__(self):
        if self.status == VerdictStatus.SAT:
            return f"VerifierResult(SAT, model={self.model})"
        if self.status in (VerdictStatus.ERROR, VerdictStatus.TIMEOUT):
            return f"VerifierResult({self.status.value.upper()}, error={self.error!r})"
        return f"VerifierResult({self.status.value.upper()})"


# Footer ejecutado DESPUES del codigo del usuario, dentro del mismo subproceso.
# Busca una variable `solver` (z3.Solver u Optimize) en el namespace global del
# codigo ejecutado, llama a check() y serializa el resultado como JSON a stdout.
# Llamar a check() aqui (en vez de confiar en que el codigo del usuario ya lo
# haya hecho) hace el protocolo robusto: no importa si el Formalizador genero
# o no esa linea, el resultado siempre se captura igual.
_RUNNER_FOOTER = """
import json as _json
import z3 as _z3

if "solver" in globals() and isinstance(globals()["solver"], (_z3.Solver, _z3.Optimize)):
    _s = globals()["solver"]
else:
    _candidates = [v for n, v in list(globals().items())
                   if isinstance(v, (_z3.Solver, _z3.Optimize)) and not n.startswith("_")]
    if not _candidates:
        raise RuntimeError("El codigo no definio ninguna variable 'solver' de tipo z3.Solver/Optimize")
    _s = _candidates[0]

_result = _s.check()
_out = {"status": str(_result)}

if str(_result) == "sat":
    _m = _s.model()
    _out["model"] = {str(d): str(_m[d]) for d in _m.decls()}

print("__VERIFIER_RESULT_START__")
print(_json.dumps(_out))
print("__VERIFIER_RESULT_END__")
"""


def run_z3_code(code: str, timeout_seconds: float = 10.0) -> VerifierResult:
    """
    Ejecuta `code` (que debe definir una variable `solver` de tipo z3.Solver
    o z3.Optimize con los asserts correspondientes) en un subproceso aislado
    de Python, con timeout. El propio wrapper llama a solver.check(), asi que
    el codigo del Formalizador no necesita (pero puede) llamarlo el mismo.

    Devuelve un VerifierResult con el veredicto (sat/unsat/unknown), el modelo
    si aplica (sat), o un error si el codigo no es valido / lanzo una excepcion
    / se paso del timeout.
    """
    full_script = code + "\n\n" + _RUNNER_FOOTER

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(full_script)
            tmp_path = tmp.name

        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as e:
            return VerifierResult(
                status=VerdictStatus.TIMEOUT,
                error=f"El codigo Z3 no termino en {timeout_seconds}s (posible loop o problema muy dificil)",
                raw_stdout=e.stdout if isinstance(e.stdout, str) else "",
                raw_stderr=e.stderr if isinstance(e.stderr, str) else "",
            )

        stdout, stderr = proc.stdout, proc.stderr

        if proc.returncode != 0:
            return VerifierResult(
                status=VerdictStatus.ERROR,
                error=stderr.strip() or f"El subproceso termino con codigo {proc.returncode}",
                raw_stdout=stdout,
                raw_stderr=stderr,
            )

        if "__VERIFIER_RESULT_START__" not in stdout or "__VERIFIER_RESULT_END__" not in stdout:
            return VerifierResult(
                status=VerdictStatus.ERROR,
                error="El subproceso no produjo el resultado esperado",
                raw_stdout=stdout,
                raw_stderr=stderr,
            )

        json_blob = stdout.split("__VERIFIER_RESULT_START__")[1].split("__VERIFIER_RESULT_END__")[0].strip()
        parsed = json.loads(json_blob)

        status_str = parsed["status"]
        if status_str == "sat":
            status = VerdictStatus.SAT
        elif status_str == "unsat":
            status = VerdictStatus.UNSAT
        else:
            status = VerdictStatus.UNKNOWN

        return VerifierResult(
            status=status,
            model=parsed.get("model"),
            raw_stdout=stdout,
            raw_stderr=stderr,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


if __name__ == "__main__":
    # --- Ejemplos hardcodeados de teoria de numeros, para probar el verifier ---

    print("=" * 70)
    print("EJEMPLO 1: para todo entero n, n^3 - n es divisible por 6 (VERDADERO)")
    print("=" * 70)
    # LIMITACION DE Z3 (importante, descubierta probando esto): la aritmetica
    # NO lineal entera (productos de variables, como n*n*n) usando el sort
    # `Int` es muy poco confiable en Z3 -- incluso acotando el rango de
    # busqueda con n >= -50 y n <= 50, el solver puede tardar mas de varios
    # segundos o devolver "unknown" en vez de resolver, porque el procedimiento
    # de decision para NIA (nonlinear integer arithmetic) no es completo y no
    # explota automaticamente que el dominio es finito.
    # La solucion practica: para busquedas ACOTADAS (rango finito conocido),
    # usar `z3.BitVec` (ancho fijo) en vez de `z3.Int`. Z3 resuelve BitVec via
    # bit-blasting + SAT, que es mucho mas eficiente para estos casos finitos.
    # Hay que elegir un ancho de bits suficiente para que no haya overflow
    # silencioso (aqui 32 bits alcanza de sobra para n*n*n con n in [-200,200]).
    code1 = """
import z3

n = z3.BitVec('n', 32)
solver = z3.Solver()
# Buscamos un CONTRAEJEMPLO: existe n en [-200,200] tal que n^3 - n NO es
# divisible por 6. Si Z3 dice UNSAT, no existe contraejemplo en ese rango.
cubo = n * n * n
solver.add(n >= -200, n <= 200)
solver.add((cubo - n) % 6 != 0)
"""
    r1 = run_z3_code(code1)
    print(r1)
    print("-> UNSAT esperado (no hay contraejemplo, la afirmacion es verdadera)\n")

    print("=" * 70)
    print("EJEMPLO 2: n^2 + n + 41 es primo para todo n >= 0 (FALSO, n=40)")
    print("=" * 70)
    # Z3 no tiene un predicado nativo "es primo" practico para enteros sin
    # acotar, asi que para detectar el contraejemplo conocido (n=40, donde
    # n^2+n+41 = 1681 = 41*41) restringimos la busqueda a un rango acotado:
    # n en [0,50] y, dado que basta buscar divisores hasta sqrt(valor), d
    # tal que d*d <= valor. Igual que en el ejemplo 1, esta version con
    # `z3.Int` (probada primero) daba timeout por la misma limitacion de
    # aritmetica no lineal entera -- usamos `z3.BitVec` (sin signo, 32 bits)
    # para que Z3 lo resuelva por bit-blasting en vez de NIA.
    code2 = """
import z3

n = z3.BitVec('n', 32)
d = z3.BitVec('d', 32)
solver = z3.Solver()

valor = n * n + n + 41
solver.add(n >= 0, n <= 50)        # busqueda acotada (Z3 no enumera "es primo" sin acotar)
solver.add(d > 1, d * d <= valor)  # basta buscar divisores hasta sqrt(valor)
solver.add(z3.URem(valor, d) == 0)  # existe un divisor propio -> valor NO es primo
"""
    r2 = run_z3_code(code2)
    print(r2)
    print("-> SAT esperado, con n=40 (y algun divisor d) como contraejemplo\n")

    print("=" * 70)
    print("EJEMPLO 3: codigo invalido (sin variable 'solver') -> debe dar ERROR")
    print("=" * 70)
    code3 = """
import z3
x = 1 + 1
"""
    r3 = run_z3_code(code3)
    print(r3)
    print("-> ERROR esperado (no se definio 'solver')\n")

    print("=" * 70)
    print("EJEMPLO 4: timeout (codigo artificialmente lento)")
    print("=" * 70)
    code4 = """
import time
time.sleep(5)
import z3
solver = z3.Solver()
solver.add(z3.Bool('x'))
"""
    r4 = run_z3_code(code4, timeout_seconds=1.0)
    print(r4)
    print("-> TIMEOUT esperado (limite de 1s, el codigo duerme 5s)\n")
