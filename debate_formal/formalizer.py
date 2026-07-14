"""
formalizer.py - El Formalizador: un agente LLM cuya UNICA responsabilidad es
traducir afirmaciones / argumentos de aritmetica y teoria de numeros, en
lenguaje natural, a codigo Python que usa z3-solver.

El Formalizador NO opina sobre si el argumento es correcto -- eso lo decide
el Verificador (Z3), ejecutando el codigo que aca se genera.

Usa Ollama (modelo local, por defecto "llama3") en vez de la API de Anthropic
para evitar costos y la necesidad de una API key -- decision tomada para este
proyecto porque el usuario ya tiene Ollama corriendo localmente. Si en el
futuro se quiere volver a un modelo mas capaz via API, alcanza con cambiar la
llamada a `ollama.chat` por una al SDK de `anthropic`; el resto del modulo
(parsing, prompts, protocolo de tags) no depende del proveedor.

DISEÑO HIBRIDO para el caso de INDUCCION (importante, ver mas abajo):
probando este modulo con llama3 8B encontramos que, a diferencia del caso
"no necesita induccion" (donde el LLM escribe codigo Z3 libre y funciona
bien), pedirle al modelo que escriba a mano el codigo Z3 del paso inductivo
es poco confiable: mezclaba z3.Int con z3.BitVec (sort mismatch), inventaba
variables nuevas sin relacion con la hipotesis, o directamente definia
funciones Python invalidas. Un modelo chico es bueno extrayendo informacion
simple de un texto, pero malo escribiendo codigo con las restricciones de
tipos que Z3 exige. Por eso, para el caso inductivo, el LLM SOLO extrae 4
valores algebraicos simples (ver `_build_induction_code`) y el codigo Z3 del
caso base y el paso inductivo se genera de forma 100% deterministica en
Python, sin que el LLM escriba una sola linea de Z3 para ese caso.
"""

import ast
import os
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

import ollama

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifier import VerdictStatus, run_z3_code

DEFAULT_MODEL = os.environ.get("FORMALIZER_MODEL", "llama3")
MAX_RETRIES = int(os.environ.get("FORMALIZER_MAX_RETRIES", "2"))


@dataclass
class FormalizationResult:
    needs_induction: bool
    code: Optional[str] = None                 # cuando needs_induction=False
    base_case_code: Optional[str] = None        # cuando needs_induction=True (generado, no escrito por el LLM)
    inductive_step_code: Optional[str] = None   # cuando needs_induction=True (generado, no escrito por el LLM)
    valor_base: Optional[str] = None            # extraido por el LLM, para transparencia/debug
    suma_en_valor_base: Optional[str] = None
    termino_recurrente: Optional[str] = None
    formula_cerrada: Optional[str] = None
    raw_response: str = ""
    parse_error: Optional[str] = None

    @property
    def ok(self) -> bool:
        if self.parse_error is not None:
            return False
        if self.needs_induction:
            return bool(self.base_case_code) and bool(self.inductive_step_code)
        return bool(self.code)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
# Notas sobre las limitaciones de Z3 que el Formalizador debe respetar
# (descubiertas empiricamente probando verifier.py en la Fase 1 y este modulo
# en la Fase 2):
#
# 1. Z3 es logica de primer orden: NO hace induccion matematica automatica.
#    Una afirmacion "para todo n, suma(1..n) = n(n+1)/2" no se puede verificar
#    pidiendole a Z3 un ForAll sin acotar sobre una formula recursiva -- hay
#    que partirla en caso base + paso inductivo.
#
# 2. La aritmetica NO LINEAL entera (productos de variables, como n*n*n o
#    n*d) usando el sort `Int` es POCO CONFIABLE en Z3, incluso acotando el
#    rango de busqueda con asserts tipo "n >= -50, n <= 50": el procedimiento
#    de decision para NIA no es completo y puede tardar mucho o devolver
#    "unknown" en casos chicos. La solucion practica es usar `z3.BitVec` (ancho
#    fijo, p.ej. 32 bits) en vez de `z3.Int` para CUALQUIER busqueda acotada
#    que involucre productos de variables -- Z3 lo resuelve por bit-blasting
#    en milisegundos.
#
# 3. Acotar solo una expresion DERIVADA (ej. "valor <= 1000") no evita el
#    overflow modular de BitVec si la variable de base (ej. "n") no tiene,
#    ella misma, cota inferior Y superior chicas: Z3 puede "encontrar" un
#    contraejemplo FALSO via wraparound. Hay que acotar cada variable BitVec
#    directamente.
#
# 4. Un LLM chico (8B) escribiendo codigo Z3 libre para el PASO INDUCTIVO es
#    poco confiable (mezcla sorts, inventa variables/funciones). Por eso ese
#    caso usa un protocolo de extraccion + generacion de codigo deterministica
#    (ver DISEÑO HIBRIDO en el docstring del modulo).
SYSTEM_PROMPT = """Sos el Formalizador de un sistema de debate matematico formal.

Tu UNICA tarea es traducir afirmaciones o argumentos de aritmetica / teoria de
numeros, escritos en lenguaje natural, a una representacion formal que despues
se verifica con Z3. NUNCA opines sobre si el argumento es correcto o no -- eso
lo decide Z3, no vos.

# Paso 1: decidi si la afirmacion necesita induccion matematica

- NO necesita induccion: afirmaciones sobre una propiedad de n que se puede
  chequear directamente (paridad, divisibilidad, primalidad, desigualdades
  algebraicas) sin razonar sobre una recurrencia.
- SI necesita induccion: formulas cerradas que dependen de una suma o
  recurrencia sobre todos los enteros hasta n (sumatorias, productorias,
  identidades tipo "1+2+...+n = n(n+1)/2").

# Caso A: NO necesita induccion

Respondé EXACTAMENTE con este formato (nada de texto ni markdown afuera):

<NEEDS_INDUCTION>false</NEEDS_INDUCTION>
<CODE>
(codigo python con z3 aca)
</CODE>

Reglas para el codigo de <CODE>:

1. SIEMPRE debe definir una variable llamada exactamente `solver`
   (instancia de z3.Solver()), con los `solver.add(...)` necesarios.
2. El unico import disponible y necesario es `import z3`. No uses ninguna
   otra libreria.
3. La forma de verificar una afirmacion con Z3 es buscando un CONTRAEJEMPLO:
   agregás la NEGACION de lo que querés probar. Si Z3 devuelve UNSAT, no
   existe contraejemplo y la afirmacion es verdadera. Si devuelve SAT, el
   modelo que devuelve Z3 ES un contraejemplo real.
4. La aritmetica no lineal entera (multiplicar dos variables, como n*n o n*d)
   usando z3.Int es poco confiable, incluso con rangos chicos acotados. Por
   eso, para CUALQUIER busqueda de contraejemplo en un rango acotado que
   involucre productos de variables, USA z3.BitVec en vez de z3.Int:
       n = z3.BitVec('n', 32)
   en vez de
       n = z3.Int('n')
   Para modulo con signo usa z3.SRem(a, b); para modulo sin signo (cuando
   sabes que ambos operandos son no negativos) usa z3.URem(a, b). Usa SIEMPRE
   un ancho de bits generoso (32 bits minimo).
5. MUY IMPORTANTE: acotar solo una expresion derivada (ej. "valor <= 1000")
   NO ALCANZA. TODA variable BitVec que participe en una multiplicacion debe
   tener, ELLA MISMA, una cota inferior Y superior explicitas con literales
   chicos (decenas, como "n >= 0, n <= 50"). Sin eso, Z3 puede devolver un
   contraejemplo FALSO por overflow modular.
6. Si la afirmacion es sobre TODOS los enteros sin acotar y es puramente
   lineal (sin productos de variables), podes usar z3.Int normal sin acotar
   rango. Si involucra productos de variables, preferi acotar con BitVec
   (regla 4) -- es mucho mas confiable que dejarlo sin acotar.
7. NUNCA uses un bucle `for`/`while` de Python que dependa del valor de una
   variable simbolica de Z3, ni operaciones de Python como `**`, `int(...)`,
   `float(...)` sobre esas variables -- son simbolos, no numeros. Si
   necesitas expresar "existe un divisor", usa una variable simbolica
   adicional (como `d` en el Ejemplo 2) y dejá que Z3 la busque.

## Ejemplo 1: "para todo entero n, n^3 - n es divisible por 6" (no necesita induccion)

<NEEDS_INDUCTION>false</NEEDS_INDUCTION>
<CODE>
import z3

n = z3.BitVec('n', 32)
solver = z3.Solver()
cubo = n * n * n
solver.add(n >= -200, n <= 200)
solver.add((cubo - n) % 6 != 0)
</CODE>

## Ejemplo 2: "n^2 + n + 41 es primo para todo n >= 0" (no necesita induccion, se espera contraejemplo)

<NEEDS_INDUCTION>false</NEEDS_INDUCTION>
<CODE>
import z3

n = z3.BitVec('n', 32)
d = z3.BitVec('d', 32)
solver = z3.Solver()
valor = n * n + n + 41
solver.add(n >= 0, n <= 50)
solver.add(d > 1, d * d <= valor)
solver.add(z3.URem(valor, d) == 0)
</CODE>

# Caso B: SI necesita induccion

NO escribas codigo Z3. En vez de eso, identificá 4 datos simples de la
afirmacion y respondé EXACTAMENTE con este formato:

<NEEDS_INDUCTION>true</NEEDS_INDUCTION>
<VALOR_BASE>(el primer n para el que vale la afirmacion, normalmente 0)</VALOR_BASE>
<SUMA_EN_VALOR_BASE>(el valor conocido de la suma/recurrencia en VALOR_BASE)</SUMA_EN_VALOR_BASE>
<TERMINO_RECURRENTE>(expresion en funcion de n: el termino que se suma para pasar de n-1 a n)</TERMINO_RECURRENTE>
<FORMULA_CERRADA>(expresion en funcion de n: la formula cerrada que se quiere verificar)</FORMULA_CERRADA>

Reglas MUY IMPORTANTES para estos 4 valores:

1. Cada expresion (TERMINO_RECURRENTE, FORMULA_CERRADA) se escribe SOLO en
   funcion de la variable `n`, usando unicamente numeros, `n`, espacios, y
   los simbolos `+ - * / ( )`. PROHIBIDO usar potencias (`**`, `^`), funciones,
   o cualquier otra variable que no sea `n`.
2. VALOR_BASE y SUMA_EN_VALOR_BASE son numeros enteros simples (ej. "0").
3. No agregues texto, explicacion, ni markdown -- SOLO los 4 tags.

## Ejemplo 3: "para todo n >= 0, 1+2+...+n = n(n+1)/2" (SI necesita induccion)

<NEEDS_INDUCTION>true</NEEDS_INDUCTION>
<VALOR_BASE>0</VALOR_BASE>
<SUMA_EN_VALOR_BASE>0</SUMA_EN_VALOR_BASE>
<TERMINO_RECURRENTE>n</TERMINO_RECURRENTE>
<FORMULA_CERRADA>n * (n + 1) / 2</FORMULA_CERRADA>

## Ejemplo 4: "para todo n >= 1, 1+3+5+...+(2n-1) = n^2" (suma de impares, SI necesita induccion)

<NEEDS_INDUCTION>true</NEEDS_INDUCTION>
<VALOR_BASE>1</VALOR_BASE>
<SUMA_EN_VALOR_BASE>1</SUMA_EN_VALOR_BASE>
<TERMINO_RECURRENTE>2 * n - 1</TERMINO_RECURRENTE>
<FORMULA_CERRADA>n * n</FORMULA_CERRADA>

Recorda: en el Caso A, SOLO el formato de <NEEDS_INDUCTION>/<CODE>. En el Caso
B, SOLO el formato de los 4 tags. Nunca mezcles ambos ni agregues texto extra."""


_TAG_RE = {
    "needs_induction": re.compile(r"<NEEDS_INDUCTION>\s*(true|false)\s*</NEEDS_INDUCTION>", re.IGNORECASE),
    "code": re.compile(r"<CODE>\s*(.*?)\s*</CODE>", re.DOTALL | re.IGNORECASE),
    "valor_base": re.compile(r"<VALOR_BASE>\s*(.*?)\s*</VALOR_BASE>", re.DOTALL | re.IGNORECASE),
    "suma_en_valor_base": re.compile(r"<SUMA_EN_VALOR_BASE>\s*(.*?)\s*</SUMA_EN_VALOR_BASE>", re.DOTALL | re.IGNORECASE),
    "termino_recurrente": re.compile(r"<TERMINO_RECURRENTE>\s*(.*?)\s*</TERMINO_RECURRENTE>", re.DOTALL | re.IGNORECASE),
    "formula_cerrada": re.compile(r"<FORMULA_CERRADA>\s*(.*?)\s*</FORMULA_CERRADA>", re.DOTALL | re.IGNORECASE),
}


def _strip_markdown_fences(code: str) -> str:
    """A veces el modelo envuelve el codigo en ```python ... ``` pese a que se
    le pidio que no lo haga. Lo limpiamos para que el codigo quede ejecutable."""
    code = code.strip()
    code = re.sub(r"^```(?:python)?\s*\n", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code.strip()


# ---------------------------------------------------------------------------
# Validacion segura de expresiones algebraicas extraidas por el LLM (Caso B)
# ---------------------------------------------------------------------------
# Las usamos para construir codigo Z3 por sustitucion de texto. NO usamos
# eval()/exec() en ningun momento -- en vez de eso, parseamos la expresion a
# un AST y verificamos a mano que el arbol entero esta compuesto SOLO por
# numeros, la variable 'n', y los operadores + - * / con parentesis. Si
# aparece cualquier otra cosa (llamada a funcion, atributo, nombre distinto
# de 'n', potencia, etc.) la rechazamos. Esto es deliberadamente mas
# restrictivo que un eval con namespace acotado: ni siquiera se ejecuta nada,
# solo se inspecciona la estructura del arbol sintactico.
_ALLOWED_NODE_TYPES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Name,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.USub, ast.UAdd, ast.Load,
)


class FormulaInvalidaError(ValueError):
    pass


def _validar_expresion(expr: str, etiqueta: str) -> str:
    expr = expr.strip()
    if not expr:
        raise FormulaInvalidaError(f"{etiqueta} esta vacio")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise FormulaInvalidaError(f"{etiqueta}={expr!r} no es una expresion aritmetica valida ({e})")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise FormulaInvalidaError(
                f"{etiqueta}={expr!r} contiene algo no permitido ({type(node).__name__}); "
                "solo se permiten numeros, 'n', y + - * / ( )"
            )
        if isinstance(node, ast.Name) and node.id != "n":
            raise FormulaInvalidaError(f"{etiqueta}={expr!r} usa la variable '{node.id}', solo se permite 'n'")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise FormulaInvalidaError(f"{etiqueta}={expr!r} contiene una constante no numerica")

    return expr


def _validar_entero(expr: str, etiqueta: str) -> str:
    expr = expr.strip()
    if not re.fullmatch(r"-?\d+", expr):
        raise FormulaInvalidaError(f"{etiqueta}={expr!r} debe ser un numero entero simple (ej. '0')")
    return expr


def _sustituir_n(expr: str, reemplazo: str) -> str:
    return re.sub(r"\bn\b", f"({reemplazo})", expr)


def _build_induction_code(valor_base: str, suma_en_base: str, termino: str, formula: str):
    """
    Genera, de forma 100% deterministica (sin LLM), el codigo Z3 del caso
    base y el paso inductivo a partir de los 4 valores extraidos. Lanza
    FormulaInvalidaError si algun valor no pasa la validacion (ese error se
    usa como feedback para que el LLM se corrija).
    """
    valor_base = _validar_entero(valor_base, "VALOR_BASE")
    suma_en_base = _validar_entero(suma_en_base, "SUMA_EN_VALOR_BASE")
    termino = _validar_expresion(termino, "TERMINO_RECURRENTE")
    formula = _validar_expresion(formula, "FORMULA_CERRADA")

    formula_en_base = _sustituir_n(formula, valor_base)
    base_case_code = f"""import z3

solver = z3.Solver()
# Caso base n={valor_base}: comparamos el valor conocido de la suma con la
# formula cerrada evaluada en n={valor_base}. Codigo generado automaticamente
# por formalizer.py a partir de los valores extraidos (no es codigo libre
# del LLM) -- ver diseno hibrido en el docstring del modulo.
suma_base = {suma_en_base}
formula_base = {formula_en_base}
solver.add(suma_base != formula_base)
"""

    termino_en_k_mas_1 = _sustituir_n(termino, "k + 1")
    formula_en_k = _sustituir_n(formula, "k")
    formula_en_k_mas_1 = _sustituir_n(formula, "k + 1")
    inductive_step_code = f"""import z3

k = z3.Int('k')
solver = z3.Solver()
solver.add(k >= 0)
# Hipotesis inductiva H: la formula cerrada vale para k (asumida cierta).
# Recurrencia: S(k+1) = S(k) + termino(k+1). Sustituyendo H (S(k) por la
# formula cerrada evaluada en k), buscamos un CONTRAEJEMPLO al paso
# inductivo: existe k que cumpla H pero rompa la formula para k+1?
# Codigo generado automaticamente por formalizer.py.
s_k = {formula_en_k}
s_k_mas_1 = s_k + ({termino_en_k_mas_1})
formula_k_mas_1 = {formula_en_k_mas_1}
solver.add(s_k_mas_1 != formula_k_mas_1)
"""
    return base_case_code, inductive_step_code


def _parse_response(raw: str) -> FormalizationResult:
    needs_induction_match = _TAG_RE["needs_induction"].search(raw)
    if not needs_induction_match:
        return FormalizationResult(
            needs_induction=False,
            raw_response=raw,
            parse_error="No se encontro el tag <NEEDS_INDUCTION> en la respuesta del modelo",
        )

    needs_induction = needs_induction_match.group(1).lower() == "true"

    if not needs_induction:
        code_match = _TAG_RE["code"].search(raw)
        if not code_match:
            return FormalizationResult(
                needs_induction=False,
                raw_response=raw,
                parse_error="needs_induction=false pero no se encontro el tag <CODE>",
            )
        return FormalizationResult(
            needs_induction=False,
            code=_strip_markdown_fences(code_match.group(1)),
            raw_response=raw,
        )

    matches = {k: _TAG_RE[k].search(raw) for k in
               ("valor_base", "suma_en_valor_base", "termino_recurrente", "formula_cerrada")}
    faltantes = [k for k, m in matches.items() if not m]
    if faltantes:
        return FormalizationResult(
            needs_induction=True,
            raw_response=raw,
            parse_error=f"needs_induction=true pero faltan tags: {faltantes}",
        )

    valor_base = matches["valor_base"].group(1).strip()
    suma_en_base = matches["suma_en_valor_base"].group(1).strip()
    termino = matches["termino_recurrente"].group(1).strip()
    formula = matches["formula_cerrada"].group(1).strip()

    try:
        base_code, step_code = _build_induction_code(valor_base, suma_en_base, termino, formula)
    except FormulaInvalidaError as e:
        return FormalizationResult(
            needs_induction=True,
            valor_base=valor_base,
            suma_en_valor_base=suma_en_base,
            termino_recurrente=termino,
            formula_cerrada=formula,
            raw_response=raw,
            parse_error=str(e),
        )

    return FormalizationResult(
        needs_induction=True,
        base_case_code=base_code,
        inductive_step_code=step_code,
        valor_base=valor_base,
        suma_en_valor_base=suma_en_base,
        termino_recurrente=termino,
        formula_cerrada=formula,
        raw_response=raw,
    )


_BITVEC_DECL_RE = re.compile(r"(\w+)\s*=\s*z3\.BitVec\(")


def _missing_bitvec_bounds(code: str) -> List[str]:
    """
    Chequeo HEURISTICO (no es un parser real, solo busca patrones de texto):
    para cada variable declarada con z3.BitVec, confirma que la variable
    APARECE, ella misma, en al menos una comparacion de cota inferior y una
    de cota superior. Ver explicacion completa en el punto 3 de las notas
    sobre limitaciones de Z3, arriba del SYSTEM_PROMPT.
    """
    faltantes = []
    for var in set(_BITVEC_DECL_RE.findall(code)):
        tiene_cota_inferior = bool(
            re.search(rf"\b{re.escape(var)}\s*>=", code)
            or re.search(rf">=\s*{re.escape(var)}\b", code)
            or re.search(rf"\b{re.escape(var)}\s*>\s*-?\d", code)
            or re.search(rf"-?\d\s*<\s*{re.escape(var)}\b", code)
        )
        tiene_cota_superior = bool(
            re.search(rf"\b{re.escape(var)}\s*<=", code)
            or re.search(rf"<=\s*{re.escape(var)}\b", code)
            or re.search(rf"\b{re.escape(var)}\s*<\s*-?\d", code)
            or re.search(rf"-?\d\s*>\s*{re.escape(var)}\b", code)
        )
        if not (tiene_cota_inferior and tiene_cota_superior):
            faltantes.append(var)
    return faltantes


def _code_smoke_test_errors(result: FormalizationResult) -> Optional[str]:
    """
    Corre el snippet de <CODE> (caso no-inductivo) a traves del Verifier para
    confirmar que (a) ejecuta sin errores de Python/Z3, y (b) pasa el chequeo
    heuristico de cotas de BitVec. El caso inductivo NO pasa por aca: su
    codigo es generado deterministicamente y ya se valida en
    `_build_induction_code` (la unica fuente de error ahi son las 4
    expresiones extraidas, no errores de sintaxis Z3).
    """
    if result.needs_induction:
        return None

    verdict = run_z3_code(result.code)
    if verdict.status == VerdictStatus.ERROR:
        return f"El bloque <CODE> fallo al ejecutarse con este error:\n{verdict.error}"

    faltantes = _missing_bitvec_bounds(result.code)
    if faltantes:
        return (
            f"El bloque <CODE> corrio sin error de Python, pero la(s) variable(s) "
            f"{faltantes} (declaradas con z3.BitVec) no tienen cota inferior Y "
            "superior EXPLICITAS sobre la variable misma (no alcanza con acotar "
            "una expresion derivada). Sin esas cotas, Z3 puede encontrar un "
            "contraejemplo FALSO por overflow modular de 32 bits."
        )
    return None


def formalize(texto: str, model: str = DEFAULT_MODEL, max_retries: int = MAX_RETRIES) -> FormalizationResult:
    """
    Traduce una afirmacion o argumento en lenguaje natural (`texto`) a una
    formalizacion Z3, llamando al LLM local via Ollama. Devuelve un
    FormalizationResult -- revisa `.ok` antes de usar `.code` /
    `.base_case_code` / `.inductive_step_code`.

    Incluye un bucle de auto-correccion: si la respuesta no sigue el formato
    de tags esperado, si las expresiones del caso inductivo no son validas, o
    si el codigo del caso no-inductivo falla al ejecutarse (error real,
    detectado corriendolo contra el Verifier), se le devuelve el error
    concreto al modelo y se le pide que se corrija, hasta `max_retries`
    veces. Esto compensa que un modelo local chico (ej. llama3 8B) no siempre
    sigue el protocolo al primer intento -- un error concreto es una señal de
    correccion mucho mas fuerte que instrucciones abstractas del prompt.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": texto},
    ]

    result = FormalizationResult(needs_induction=False, parse_error="no se intento ninguna llamada")

    for attempt in range(max_retries + 1):
        response = ollama.chat(model=model, messages=messages, options={"temperature": 0})
        raw = response["message"]["content"]
        result = _parse_response(raw)

        if attempt == max_retries:
            break

        if result.parse_error is not None:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Hubo un problema con tu respuesta: {result.parse_error}. "
                    "Recorda: SOLO los tags del formato correspondiente, sin "
                    "texto extra ni markdown. Reintenta."
                ),
            })
            continue

        smoke_error = _code_smoke_test_errors(result)
        if smoke_error is not None:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"{smoke_error}\n\nCorregi el codigo. Recorda seguir el patron "
                    "exacto de los ejemplos, no inventes funciones ni uses bucles "
                    "de Python sobre variables simbolicas de Z3. Respondé de nuevo "
                    "con el mismo formato de tags, codigo completo."
                ),
            })
            continue

        break

    return result


if __name__ == "__main__":
    claims = [
        "Para todo entero n, n al cubo menos n es divisible por 6.",
        "n al cuadrado mas n mas 41 es un numero primo para todo n mayor o igual a 0.",
        "Para todo n mayor o igual a 0, la suma 1+2+...+n es igual a n(n+1)/2.",
    ]

    for i, claim in enumerate(claims, 1):
        print("=" * 70)
        print(f"AFIRMACION {i}: {claim}")
        print("=" * 70)

        result = formalize(claim)

        if not result.ok:
            print(f"[FORMALIZER] parse_error={result.parse_error}")
            print("--- raw response ---")
            print(result.raw_response)
            print()
            continue

        if not result.needs_induction:
            print("[FORMALIZER] no requiere induccion. Codigo generado:")
            print(result.code)
            print()
            verdict = run_z3_code(result.code)
            print(f"[VERIFIER] {verdict}")
        else:
            print(
                "[FORMALIZER] requiere induccion. Extraido: "
                f"VALOR_BASE={result.valor_base!r} "
                f"SUMA_EN_VALOR_BASE={result.suma_en_valor_base!r} "
                f"TERMINO_RECURRENTE={result.termino_recurrente!r} "
                f"FORMULA_CERRADA={result.formula_cerrada!r}"
            )
            print("--- CASO BASE (codigo generado, no escrito por el LLM) ---")
            print(result.base_case_code)
            base_verdict = run_z3_code(result.base_case_code)
            print(f"[VERIFIER] caso base -> {base_verdict}")
            print("--- PASO INDUCTIVO (codigo generado, no escrito por el LLM) ---")
            print(result.inductive_step_code)
            step_verdict = run_z3_code(result.inductive_step_code)
            print(f"[VERIFIER] paso inductivo -> {step_verdict}")
        print()
