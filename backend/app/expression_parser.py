"""
Parser модул - претворa суров текст (од OCR или рачен внес) во
математички објект што SymPy може да го разбере (Expr или Eq).

Ова е слојот помеѓу "суров стринг" и "математичка структура" -
чекор 2 од архитектурата (види README).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sympy import Eq
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Чести замени - Unicode математички симболи, симболи што Tesseract
# понекогаш ги препознава наместо стандардните ASCII оператори,
# и чести OCR грешки за математички изрази.
_SYMBOL_REPLACEMENTS = {
    "×": "*",
    "·": "*",
    "÷": "/",
    "−": "-",  # unicode minus
    "—": "-",  # em dash
    "–": "-",  # en dash
    "√": "sqrt",
    "π": "pi",
    "²": "**2",
    "³": "**3",
    ",": ".",  # 3,5 -> 3.5 (европски децимален запис)
}


class ParseError(ValueError):
    """Изразот не можеше да се разбере/парсира."""


# Дозволени карактери во нормализиран израз - сè останато (пр. "&", "|",
# "~", "\", "{", "}", "#") се третира како грешка во препознавањето, не
# се проследува понатаму кон sympy parse_expr (кое инаку може "тивко" да
# ги прифати некои од овие како валидни Python оператори).
_ALLOWED_CHARS_RE = re.compile(r"[0-9a-zA-Z+\-*/^.,=()<>]*")


# ---------------------------------------------------------------------------
# LaTeX -> plain текст (за излезот од специјализираниот TrOCR_Math_handwritten
# модел, кој враќа LaTeX наместо обичен ASCII израз). Не користиме
# latex2sympy2 - пакетот е скршен на овој систем (antlr4 верзиски конфликт) -
# затоа рачно ги нормализираме најчестите LaTeX конструкции за основна
# алгебра, а остатокот минува низ normalize_text() како и обично.
# ---------------------------------------------------------------------------

_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_BRACED_POWER_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_SUBSCRIPT_RE = re.compile(r"_\s*\{?[^{}\s]*\}?")  # индекси - ги отфрламе (вон опфат)

# \begin{matrix}...\end{matrix} (и pmatrix/bmatrix/array итн.) - моделот
# понекогаш "гледа" вишок ред (пр. линија од хартијата, сенка) и го враќа
# изразот завиткан во multi-row конструкција. За основна алгебра земаме
# само единствениот непразен ред.
_ENV_RE = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}(?:\{[^{}]*\})?")


def _strip_matrix_rows(text: str) -> str:
    text = _ENV_RE.sub("", text)
    if "\\\\" in text:
        rows = [r.strip() for r in text.split("\\\\")]
        rows = [r for r in rows if r]
        if rows:
            text = rows[0]
    return text

_LATEX_REPLACEMENTS = {
    r"\left": "",
    r"\right": "",
    r"\times": "*",
    r"\cdot": "*",
    r"\div": "/",
    r"\pi": "pi",
    r"\,": "",
    r"\;": "",
    r"\!": "",
    "$": "",
}


def latex_to_plain(latex: str) -> str:
    """Претвора (основен подмножество) LaTeX во ASCII израз што
    normalize_text()/parse_expr() можат да го разберат."""
    text = latex.strip()
    text = _strip_matrix_rows(text)

    # \frac{a}{b} -> (a)/(b) - примени повеќепати за да фатиш неколку
    # дропки во истиот израз (без вгнездени дропки - вон опфат на MVP)
    prev = None
    while prev != text:
        prev = text
        text = _FRAC_RE.sub(r"(\1)/(\2)", text)

    text = _SQRT_RE.sub(r"sqrt(\1)", text)
    text = _BRACED_POWER_RE.sub(r"^(\1)", text)
    text = _SUBSCRIPT_RE.sub("", text)

    for bad, good in _LATEX_REPLACEMENTS.items():
        text = text.replace(bad, good)

    text = text.replace("{", "(").replace("}", ")")
    return text


@dataclass
class ParsedProblem:
    raw_text: str
    normalized_text: str
    kind: str  # "equation" | "expression"
    sympy_obj: object  # Eq(...) или Expr
    variables: list = field(default_factory=list)


def normalize_text(raw: str) -> str:
    """Чисти суров текст (од OCR или рачен внес) во форма што SymPy
    парсерот може да ја обработи."""
    text = raw.strip()

    for bad, good in _SYMBOL_REPLACEMENTS.items():
        text = text.replace(bad, good)

    # Отстрани празни места - "2 x + 3" -> "2x+3"
    text = "".join(text.split())

    # sqrt без загради, пр. "sqrt9" -> "sqrt(9)" - чест случај кога OCR
    # го изгуби заградата или корисникот ја испуштил
    text = re.sub(r"sqrt(\d+(\.\d+)?)", r"sqrt(\1)", text)
    text = re.sub(r"sqrt([a-zA-Z])(?!\()", r"sqrt(\1)", text)

    if not text:
        raise ParseError("Празен израз - нема што да се препознае/парсира.")

    # Строга валидација - отфрли какви било карактери надвор од очекуваната
    # математичка азбука ПРЕД да стигне до parse_expr(). Ова е важно бидејќи
    # sympy/Python инаку "тивко" би прифатил невалидни карактери како
    # валидни оператори (пр. "&" = bitwise-AND во Python) и би пресметал
    # лажно веродостоен, но целосно погрешен резултат наместо грешка -
    # реален случај забележан од (недоволно исчистен) LaTeX излез од
    # handwriting моделот кога содржи стрски matrix-артефакти ("&").
    if not _ALLOWED_CHARS_RE.fullmatch(text):
        bad_chars = sorted(set(re.sub(r"[0-9a-zA-Z+\-*/^.,=()<>]", "", text)))
        raise ParseError(
            f"Изразот содржи непрепознаени карактери ({''.join(bad_chars)}) - "
            f"веројатно грешка во препознавањето. Провери/поправи го текстот."
        )

    return text


def parse(raw_text: str) -> ParsedProblem:
    """
    Главна функција: суров текст -> ParsedProblem со sympy објект.

    Поддржува равенки ("2x+3=11") и обични изрази за симплификација
    ("2x+3(x-1)").
    """
    normalized = normalize_text(raw_text)

    # Дозволуваме само еден "=" (равенка). Повеќе од еден "=" не е
    # валидна равенка во овој MVP опсег.
    if normalized.count("=") > 1:
        raise ParseError("Изразот содржи повеќе од еден знак '=' - не можам да го решам.")

    try:
        if "=" in normalized:
            lhs_str, rhs_str = normalized.split("=")
            lhs = parse_expr(lhs_str, transformations=_TRANSFORMATIONS)
            rhs = parse_expr(rhs_str, transformations=_TRANSFORMATIONS)
            sympy_obj = Eq(lhs, rhs)
            kind = "equation"
        else:
            sympy_obj = parse_expr(normalized, transformations=_TRANSFORMATIONS)
            kind = "expression"
    except Exception as exc:
        # SymPy/Python-тен tokenizer-от може да фрли разни типови грешки на
        # "смет" (garbage) внес - SympifyError, SyntaxError, TokenError,
        # AttributeError, RecursionError итн. Сите ги третираме исто: тоа
        # е неуспешен parse на корисничкиот/OCR внес, не bug во апликацијата.
        raise ParseError(
            f"Не можам да го парсирам изразот '{normalized}'. "
            f"Провери дали е точно препознаен/внесен."
        ) from exc

    free_symbols = sorted(sympy_obj.free_symbols, key=lambda s: s.name)

    return ParsedProblem(
        raw_text=raw_text,
        normalized_text=normalized,
        kind=kind,
        sympy_obj=sympy_obj,
        variables=free_symbols,
    )


def parse_latex(latex: str) -> ParsedProblem:
    """Convenience: LaTeX израз (пр. излез од TrOCR_Math_handwritten) ->
    ParsedProblem. За raw_text го чуваме оригиналниот LaTeX (не plain
    верзијата) за да остане видливо што точно препознал моделот."""
    plain = latex_to_plain(latex)
    parsed = parse(plain)
    return ParsedProblem(
        raw_text=latex,
        normalized_text=parsed.normalized_text,
        kind=parsed.kind,
        sympy_obj=parsed.sympy_obj,
        variables=parsed.variables,
    )
