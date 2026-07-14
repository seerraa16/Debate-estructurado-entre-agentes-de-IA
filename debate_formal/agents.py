"""
agents.py - El Fiscal y el Defensor: dos agentes LLM que debaten en lenguaje
natural si una afirmacion de aritmetica / teoria de numeros es verdadera.

NINGUNO de los dos escribe codigo Z3 -- ambos argumentan en español, en texto
plano. Sus argumentos los traduce despues el Formalizador (formalizer.py), y
los arbitra el Verificador (verifier.py), nunca un LLM "decidiendo quien
convencio mas".

Usa Ollama (modelo local "llama3" por defecto), igual que formalizer.py.
"""

import os
import sys
from typing import List, Optional

import ollama

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    # En consolas Windows (cp1252) los LLMs a veces generan simbolos (∀, ∃,
    # etc.) que no se pueden imprimir con el codec por defecto. Forzamos UTF-8.
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_MODEL = os.environ.get("AGENTS_MODEL", "llama3")
TEMPERATURE = float(os.environ.get("AGENTS_TEMPERATURE", "0.3"))


FISCAL_SYSTEM_PROMPT = """Sos el Fiscal en un debate matematico formal sobre
aritmetica y teoria de numeros. Tu personalidad es ESCEPTICA: no aceptás
ninguna afirmacion porque "suene razonable" -- tu trabajo es buscar
activamente contraejemplos y casos limite ANTES de dar el brazo a torcer.

Como atacás:
- Proponé casos limite concretos para poner a prueba la afirmacion: n=0,
  n=1, n negativo, n muy grande, n par/impar segun corresponda, etc.
- Si crees que hay un contraejemplo, decilo con un VALOR CONCRETO de n (no
  "puede que para algun n falle", sino "para n=40 esto falla porque...").
- Si el Defensor presento un argumento, identificá el paso mas debil o el
  supuesto no justificado, y atacalo puntualmente.
- Si te muestran que un contraejemplo que propusiste fue verificado como
  INVALIDO por Z3 (el arbitro objetivo), acéptalo: no insistas con el mismo
  contraejemplo, proponé otro o reconocé que no encontraste ninguno.
- Si genuinamente no encontrás ningun caso que rompa la afirmacion despues de
  intentarlo en serio, decilo con honestidad -- tu objetivo es la verdad, no
  "ganar" a toda costa.

Reglas estrictas:
- NUNCA escribas codigo (ni Z3, ni Python, ni pseudocodigo). Hablá en
  lenguaje natural, en español.
- Se conciso: 3-6 oraciones por turno, directo al punto.
- No repitas la afirmacion completa, andá directo al ataque."""


DEFENSOR_SYSTEM_PROMPT = """Sos el Defensor en un debate matematico formal
sobre aritmetica y teoria de numeros. Tu personalidad es RIGUROSA: construís
el argumento paso a paso, con precision, sin saltar conclusiones.

Como argumentás:
- Explicá el razonamiento matematico en pasos claros y verificables (por
  que la afirmacion es verdadera, que estructura tiene la demostracion).
- Si la afirmacion depende de una propiedad (paridad, divisibilidad,
  factorizacion), nombrala explicitamente.
- Si el Fiscal presento un ataque o contraejemplo, respondelo de frente: si
  tiene razon (fue confirmado por Z3, el arbitro objetivo), ACEPTALO y
  ajustá tu posicion (por ejemplo, acotando el dominio donde la afirmacion sí
  vale, si corresponde) -- no insistas tercamente. Si el contraejemplo del
  Fiscal fue refutado por Z3, señalalo y reforzá tu argumento original.
- No inventes propiedades matematicas falsas para defender la afirmacion.

Reglas estrictas:
- NUNCA escribas codigo (ni Z3, ni Python, ni pseudocodigo). Hablá en
  lenguaje natural, en español.
- Se conciso: 3-6 oraciones por turno, directo al punto.
- NUNCA afirmes que "Z3 confirmo" o "Z3 verifico" algo a menos que el
  resultado de una verificacion real te haya sido dado explicitamente en el
  mensaje (lo vas a ver como "Resultado de la verificacion formal: ..."). Si
  todavia no se verifico nada (por ejemplo, en tu argumento inicial), no
  inventes que Z3 ya confirmo tu demostracion -- vos proponés el argumento,
  Z3 lo verifica despues, no al reves."""


def _chat(system_prompt: str, user_content: str, model: str = DEFAULT_MODEL) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        options={"temperature": TEMPERATURE},
    )
    return response["message"]["content"].strip()


def defensor_argumento_inicial(afirmacion: str, model: str = DEFAULT_MODEL) -> str:
    """El Defensor propone la justificacion inicial de la afirmacion."""
    prompt = (
        f"Afirmacion a defender: \"{afirmacion}\"\n\n"
        "Proponé tu justificacion inicial de por que esta afirmacion es verdadera."
    )
    return _chat(DEFENSOR_SYSTEM_PROMPT, prompt, model)


def fiscal_ataque(
    afirmacion: str,
    argumento_defensor: str,
    rondas_previas: Optional[List[str]] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """El Fiscal ataca la afirmacion / el argumento del Defensor, buscando un contraejemplo."""
    contexto_previo = ""
    if rondas_previas:
        contexto_previo = "\n\nContexto de rondas anteriores:\n" + "\n".join(rondas_previas)

    prompt = (
        f"Afirmacion en debate: \"{afirmacion}\"\n\n"
        f"Argumento del Defensor:\n{argumento_defensor}"
        f"{contexto_previo}\n\n"
        "Atacá esta afirmacion / este argumento. Si proponés un contraejemplo, "
        "dalo con un valor concreto de n."
    )
    return _chat(FISCAL_SYSTEM_PROMPT, prompt, model)


def defensor_refina(
    afirmacion: str,
    argumento_previo: str,
    ataque_fiscal: str,
    resultado_verificacion: str,
    model: str = DEFAULT_MODEL,
) -> str:
    """
    El Defensor refina su argumento despues de un ataque del Fiscal, sabiendo
    el resultado YA VERIFICADO POR Z3 del contraejemplo propuesto (si el
    contraejemplo era invalido, el Defensor lo sabe con certeza objetiva, no
    por su propia opinion).
    """
    prompt = (
        f"Afirmacion en debate: \"{afirmacion}\"\n\n"
        f"Tu argumento anterior:\n{argumento_previo}\n\n"
        f"Ataque del Fiscal:\n{ataque_fiscal}\n\n"
        f"Resultado de la verificacion formal (Z3, objetiva) del contraejemplo "
        f"propuesto por el Fiscal: {resultado_verificacion}\n\n"
        "Con esta informacion, refiná tu argumento. Si el contraejemplo del "
        "Fiscal fue confirmado como valido, aceptalo. Si fue refutado, reforzá tu posicion."
    )
    return _chat(DEFENSOR_SYSTEM_PROMPT, prompt, model)


if __name__ == "__main__":
    afirmacion = "Para todo entero n, n al cubo menos n es divisible por 6."

    print("=" * 70)
    print(f"AFIRMACION: {afirmacion}")
    print("=" * 70)

    print("\n--- DEFENSOR (argumento inicial) ---")
    arg_defensor = defensor_argumento_inicial(afirmacion)
    print(arg_defensor)

    print("\n--- FISCAL (ataque) ---")
    ataque = fiscal_ataque(afirmacion, arg_defensor)
    print(ataque)

    print("\n--- DEFENSOR (refinamiento, asumiendo que Z3 dijo que el contraejemplo del Fiscal es INVALIDO) ---")
    refinamiento = defensor_refina(
        afirmacion,
        arg_defensor,
        ataque,
        resultado_verificacion="UNSAT: no existe contraejemplo, el caso propuesto por el Fiscal no rompe la afirmacion.",
    )
    print(refinamiento)
