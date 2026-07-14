"""
orchestrator.py - Maquina de estados que orquesta el debate formal entre el
Fiscal y el Defensor, usando al Formalizador para traducir argumentos a Z3 y
al Verificador (Z3) como arbitro final, objetivo.

Logica de rondas:
  0. (una sola vez) Se formaliza la AFIRMACION ORIGINAL desnuda (sin la
     narrativa de ningun agente) -> Formalizador traduce -> Verificador
     chequea. Esto sirve de chequeo base, equivalente a verificar la
     "justificacion inicial" de que la afirmacion es verdadera.
  1. El Defensor presenta su argumento en lenguaje natural (valor narrativo /
     de debate -- ver nota de diseño abajo).
  2. El Fiscal ataca (intenta encontrar un contraejemplo o fallo) ->
     Formalizador traduce el contraejemplo propuesto -> Verificador chequea
     si realmente rompe la afirmacion.
  3. Si el chequeo base o el del Fiscal produjo un contraejemplo CONFIRMADO
     por Z3: la afirmacion es FALSA, fin del debate.
  4. Si no, el Defensor refina su argumento en otra ronda (maximo 3 rondas) y
     el Fiscal vuelve a atacar.
  5. Veredicto final: lo decide el Verificador (Z3), NUNCA un LLM evaluando
     "quien convencio mas". Si Z3 nunca llega a una conclusion clara (UNKNOWN
     persistente, o errores de formalizacion), se reporta como INCONCLUSA --
     no se fuerza un veredicto.

NOTA DE DISEÑO IMPORTANTE (encontrada probando este modulo con llama3 8B):
en un primer diseño, CADA ronda se formalizaba el argumento del Defensor
(afirmacion + su razonamiento especifico de esa ronda). Esto causo DOS
falsos positivos de "FALSA" reales: el Formalizador, al tratar de seguir el
razonamiento elaborado del Defensor (que a veces se desviaba a sub-pruebas
parciales, como "primero veo divisibilidad por 2, despues por 3"), termino
generando codigo Z3 que NO formalizaba la afirmacion completa -- una vez
solo verifico paridad (irrelevante para primalidad), otra vez literalmente
PERDIO el termino "- n" de "n^3 - n" al tratar de imitar el caso-por-caso del
argumento. En ambos casos Z3 encontro un "contraejemplo" SAT que en realidad
no refutaba la afirmacion original, solo una formula distinta y mal armada.
La causa raiz: pedirle a un LLM chico que formalice una narrativa elaborada
y cambiante es mucho mas fragil que pedirle que formalice la afirmacion
desnuda (que es justamente el caso que probamos exhaustivamente en la Fase 2
y resulto confiable). La solucion: formalizar la afirmacion ORIGINAL una
sola vez al principio (desacoplada de la narrativa de cualquier agente), y
reusar ese resultado como el chequeo de fondo del debate. El Defensor sigue
argumentando en lenguaje natural en cada ronda (tiene valor real: explica el
razonamiento, reacciona a los ataques), pero esa narrativa NUNCA vuelve a
pasar por el Formalizador -- solo los ataques especificos del Fiscal lo
hacen, porque ahi SI es el rol del Fiscal proponer casos concretos nuevos
cada ronda, y en la practica resulto mucho mas confiable formalizando esos
ataques puntuales que formalizando demostraciones completas.
"""

import os
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents import defensor_argumento_inicial, defensor_refina, fiscal_ataque
from formalizer import FormalizationResult, formalize
from verifier import VerdictStatus, VerifierResult, run_z3_code

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

MAX_RONDAS = int(os.environ.get("ORCHESTRATOR_MAX_RONDAS", "3"))


@dataclass
class RoundLog:
    numero: int
    arg_defensor: str
    ataque_fiscal: str
    formalizacion_fiscal: FormalizationResult
    veredicto_fiscal: Dict[str, VerifierResult]
    clasificacion_fiscal: str
    explicacion_fiscal: str


@dataclass
class ResultadoDebate:
    afirmacion: str
    veredicto: str  # "VERDADERA" | "FALSA" | "INCONCLUSA"
    explicacion: str
    contraejemplo: Optional[dict]
    formalizacion_base: Optional[FormalizationResult] = None
    veredicto_base: Optional[Dict[str, VerifierResult]] = None
    rondas: List[RoundLog] = field(default_factory=list)


def _formalizar_y_verificar(texto: str) -> Tuple[FormalizationResult, Dict[str, VerifierResult]]:
    fr = formalize(texto)
    if not fr.ok:
        return fr, {}
    if not fr.needs_induction:
        return fr, {"code": run_z3_code(fr.code)}
    return fr, {"base": run_z3_code(fr.base_case_code), "step": run_z3_code(fr.inductive_step_code)}


def _interpretar(fr: FormalizationResult, verdicts: Dict[str, VerifierResult]) -> Tuple[str, Optional[dict], str]:
    """
    Traduce el resultado crudo de Z3 a una clasificacion de alto nivel:
    "contraejemplo_encontrado" | "sin_contraejemplo" | "inconcluso" | "error_formalizacion".
    NUNCA decide un LLM aca -- esto es interpretacion mecanica de lo que dijo Z3.
    """
    if not fr.ok:
        return "error_formalizacion", None, f"El Formalizador no pudo traducir esto a Z3: {fr.parse_error}"

    if not fr.needs_induction:
        v = verdicts["code"]
        if v.status == VerdictStatus.SAT:
            return "contraejemplo_encontrado", v.model, f"Z3 encontro un contraejemplo real: {v.model}"
        if v.status == VerdictStatus.UNSAT:
            return "sin_contraejemplo", None, "Z3 no encontro ningun contraejemplo (UNSAT)."
        return "inconcluso", None, f"Z3 no pudo llegar a una conclusion clara (status={v.status.value}, {v.error})"

    base_v, step_v = verdicts["base"], verdicts["step"]
    if base_v.status == VerdictStatus.SAT:
        return "contraejemplo_encontrado", base_v.model, "Z3 encontro un contraejemplo en el CASO BASE de la induccion."
    if step_v.status == VerdictStatus.SAT:
        return "contraejemplo_encontrado", step_v.model, "Z3 encontro un contraejemplo en el PASO INDUCTIVO."
    if base_v.status == VerdictStatus.UNSAT and step_v.status == VerdictStatus.UNSAT:
        return "sin_contraejemplo", None, "Z3 confirmo el caso base y el paso inductivo (UNSAT en ambos)."
    return (
        "inconcluso",
        None,
        f"Z3 no llego a una conclusion clara (caso base={base_v.status.value}, paso inductivo={step_v.status.value}).",
    )


# ---------------------------------------------------------------------------
# Impresion legible en terminal
# ---------------------------------------------------------------------------

def _sep(titulo: str = "") -> None:
    print("\n" + "=" * 78)
    if titulo:
        print(titulo)
        print("=" * 78)


def _imprimir_turno(agente: str, texto: str) -> None:
    print(f"\n--- {agente} ---")
    print(texto)


def _imprimir_formalizacion(etiqueta: str, fr: FormalizationResult) -> None:
    print(f"\n[FORMALIZADOR] ({etiqueta})")
    if not fr.ok:
        print(f"  No se pudo formalizar: {fr.parse_error}")
        return
    if not fr.needs_induction:
        print("  Codigo Z3 generado:")
        print(textwrap.indent(fr.code.strip(), "    "))
    else:
        print(
            "  Requiere induccion. Extraido: "
            f"VALOR_BASE={fr.valor_base!r} SUMA_EN_VALOR_BASE={fr.suma_en_valor_base!r} "
            f"TERMINO_RECURRENTE={fr.termino_recurrente!r} FORMULA_CERRADA={fr.formula_cerrada!r}"
        )
        print("  Codigo del caso base (generado):")
        print(textwrap.indent(fr.base_case_code.strip(), "    "))
        print("  Codigo del paso inductivo (generado):")
        print(textwrap.indent(fr.inductive_step_code.strip(), "    "))


def _imprimir_veredicto(etiqueta: str, verdicts: Dict[str, VerifierResult], clasificacion: str, explicacion: str) -> None:
    if "code" in verdicts:
        print(f"[VERIFICADOR] ({etiqueta}) -> {verdicts['code']}")
    elif "base" in verdicts:
        print(f"[VERIFICADOR] ({etiqueta}) caso base -> {verdicts['base']}")
        print(f"[VERIFICADOR] ({etiqueta}) paso inductivo -> {verdicts['step']}")
    print(f"[VEREDICTO PARCIAL] ({etiqueta}) {clasificacion}: {explicacion}")


def _imprimir_resultado_final(resultado: ResultadoDebate) -> None:
    _sep("VEREDICTO FINAL (decidido por Z3, no por un LLM)")
    print(f"Afirmacion: {resultado.afirmacion}")
    print(f"Veredicto: {resultado.veredicto}")
    print(f"Explicacion: {resultado.explicacion}")
    if resultado.contraejemplo:
        print(f"Contraejemplo: {resultado.contraejemplo}")


# ---------------------------------------------------------------------------
# Orquestacion del debate
# ---------------------------------------------------------------------------

def run_debate(afirmacion: str, max_rondas: int = MAX_RONDAS) -> ResultadoDebate:
    _sep(f"DEBATE FORMAL SOBRE: {afirmacion}")

    # --- Paso 0: formalizar la afirmacion desnuda UNA SOLA VEZ ---------------
    _sep("CHEQUEO BASE (afirmacion original, sin narrativa de ningun agente)")
    texto_base = (
        f"Afirmacion a verificar: \"{afirmacion}\"\n\n"
        "Formalizá esta afirmacion buscando un contraejemplo, de la forma mas "
        "simple posible (segui el patron de los ejemplos)."
    )
    fr_base, verdicts_base = _formalizar_y_verificar(texto_base)
    _imprimir_formalizacion("afirmacion original", fr_base)
    clasif_base, contraej_base, expl_base = _interpretar(fr_base, verdicts_base)
    _imprimir_veredicto("afirmacion original", verdicts_base, clasif_base, expl_base)

    if clasif_base == "contraejemplo_encontrado":
        resultado = ResultadoDebate(
            afirmacion=afirmacion,
            veredicto="FALSA",
            explicacion=f"El chequeo formal de la afirmacion original revelo un contraejemplo real: {expl_base}",
            contraejemplo=contraej_base,
            formalizacion_base=fr_base,
            veredicto_base=verdicts_base,
        )
        _imprimir_resultado_final(resultado)
        return resultado

    hubo_inconcluso = clasif_base in ("inconcluso", "error_formalizacion")

    rondas: List[RoundLog] = []
    historial_resumen: List[str] = []
    arg_defensor = None
    ataque_fiscal_anterior = None
    explicacion_fiscal_anterior = None

    for numero in range(1, max_rondas + 1):
        _sep(f"RONDA {numero}/{max_rondas}")

        if numero == 1:
            arg_defensor = defensor_argumento_inicial(afirmacion)
        else:
            arg_defensor = defensor_refina(
                afirmacion, arg_defensor, ataque_fiscal_anterior, explicacion_fiscal_anterior
            )
        _imprimir_turno("DEFENSOR", arg_defensor)

        ataque_fiscal = fiscal_ataque(afirmacion, arg_defensor, historial_resumen or None)
        _imprimir_turno("FISCAL", ataque_fiscal)

        texto_fiscal = (
            f"Afirmacion ORIGINAL Y COMPLETA a verificar (formalizá la busqueda de "
            f"contraejemplo para EXACTAMENTE esta afirmacion, nada mas): \"{afirmacion}\"\n\n"
            f"El Fiscal sostiene que la afirmacion es FALSA por lo siguiente: {ataque_fiscal}\n\n"
            "Formalizá una busqueda de contraejemplo a la afirmacion ORIGINAL completa (NO "
            "a una sub-propiedad parcial que el Fiscal haya mencionado de pasada -- por "
            "ejemplo, si la afirmacion es sobre PRIMALIDAD, formalizá la busqueda de un "
            "divisor propio como en el Ejemplo 2, NO formalices solo paridad o divisibilidad "
            "por un numero especifico aunque el Fiscal se haya enfocado en eso). Si el "
            "Fiscal menciono un valor concreto de n, usalo para acotar el rango de "
            "busqueda; si no, usa un rango razonable."
        )
        fr_fiscal, verdicts_fiscal = _formalizar_y_verificar(texto_fiscal)
        _imprimir_formalizacion("ataque del Fiscal", fr_fiscal)
        clasif_fiscal, contraej_fiscal, expl_fiscal = _interpretar(fr_fiscal, verdicts_fiscal)
        _imprimir_veredicto("ataque del Fiscal", verdicts_fiscal, clasif_fiscal, expl_fiscal)

        if clasif_fiscal in ("inconcluso", "error_formalizacion"):
            hubo_inconcluso = True

        rondas.append(RoundLog(
            numero=numero,
            arg_defensor=arg_defensor,
            ataque_fiscal=ataque_fiscal,
            formalizacion_fiscal=fr_fiscal,
            veredicto_fiscal=verdicts_fiscal,
            clasificacion_fiscal=clasif_fiscal,
            explicacion_fiscal=expl_fiscal,
        ))

        if clasif_fiscal == "contraejemplo_encontrado":
            resultado = ResultadoDebate(
                afirmacion=afirmacion,
                veredicto="FALSA",
                explicacion=f"El Fiscal encontro un contraejemplo, confirmado por Z3: {expl_fiscal}",
                contraejemplo=contraej_fiscal,
                formalizacion_base=fr_base,
                veredicto_base=verdicts_base,
                rondas=rondas,
            )
            _imprimir_resultado_final(resultado)
            return resultado

        historial_resumen.append(
            f"Ronda {numero}: Defensor argumento '{arg_defensor[:120]}...'; "
            f"Fiscal ataco '{ataque_fiscal[:120]}...'; Z3 dijo: {expl_fiscal}"
        )
        ataque_fiscal_anterior = ataque_fiscal
        explicacion_fiscal_anterior = expl_fiscal

    # Se agotaron las rondas sin que el Fiscal (ni el chequeo base) produjera
    # un contraejemplo confirmado.
    if hubo_inconcluso:
        resultado = ResultadoDebate(
            afirmacion=afirmacion,
            veredicto="INCONCLUSA",
            explicacion=(
                f"Tras {max_rondas} rondas, el Fiscal no logro un contraejemplo confirmado, "
                "pero Z3 tampoco pudo llegar a una conclusion clara en al menos un chequeo "
                "(UNKNOWN, timeout, o fallo de formalizacion). No se fuerza un veredicto."
            ),
            contraejemplo=None,
            formalizacion_base=fr_base,
            veredicto_base=verdicts_base,
            rondas=rondas,
        )
    else:
        resultado = ResultadoDebate(
            afirmacion=afirmacion,
            veredicto="VERDADERA",
            explicacion=(
                f"Tras {max_rondas} rondas, ningun contraejemplo propuesto por el Fiscal (ni el "
                "chequeo base de la afirmacion) fue confirmado por Z3 -- todos los chequeos dieron UNSAT."
            ),
            contraejemplo=None,
            formalizacion_base=fr_base,
            veredicto_base=verdicts_base,
            rondas=rondas,
        )

    _imprimir_resultado_final(resultado)
    return resultado


if __name__ == "__main__":
    run_debate("Para todo entero n, n al cubo menos n es divisible por 6.", max_rondas=1)
