"""
Parser модул - претворa суров текст (од OCR или рачен внес) во
математички објект што SymPy може да го разбере (Expr или Eq).

Ова е слојот помеѓу "суров стринг" и "математичка структура" -
чекор 2 од архитектурата (види README).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sympy import Eq, Symbol, cos, pi, sin, tan
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
}

# "," само помеѓу цифри е европски децимален запис ("3,5" -> "3.5") -
# НЕ смее да биде blanket замена бидејќи "," исто така се користи како
# разделувач на аргументи (пр. "diff(x^2,x)", "x+y=5;x-y=1" системи).
_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")


class ParseError(ValueError):
    """Изразот не можеше да се разбере/парсира."""


# Дозволени карактери во нормализиран израз - сè останато (пр. "&", "|",
# "~", "\", "{", "}", "#") се третира како грешка во препознавањето, не
# се проследува понатаму кон sympy parse_expr (кое инаку може "тивко" да
# ги прифати некои од овие како валидни Python оператори). ";" е дозволен
# бидејќи се користи само како разделувач на систем равенки (никогаш не
# стигнува до parse_expr самиот - се дели пред тоа).
_ALLOWED_CHARS_RE = re.compile(r"[0-9a-zA-Z+\-*/^.,=()<>;]*")

# diff(израз, променлива) / integrate(израз, променлива) - препознаени
# ПРЕД генеричкото equation/expression парсирање, за да можеме да го
# сочуваме оригиналниот израз (не веднаш-пресметаниот derivative/integral)
# и да генерираме чекор-по-чекор наратив во steps.py.
_DIFF_CALL_RE = re.compile(r"^diff\((.+),([a-zA-Z]\w*)\)$")
_INTEGRATE_CALL_RE = re.compile(r"^integrate\((.+),([a-zA-Z]\w*)\)$")


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

# "Козметички" LaTeX wrapper-команди - апликацијата никогаш не поддржува
# нивното значење (стил на буква, множество, хоризонтална линија итн.),
# па секое појавување е гарантирано OCR артефакт. Два реални забележани
# случаи: \overline{x+y-7;x-y=1} (систем равенки) и \mathbb{x^3-...=0}
# (кубна равенка) - истиот образец \команда{содржина}, само содржината
# е важна, wrapper-от се фрла. Само "одвиткуваме" (не бришеме) - за
# разлика од matrix-случајот, тука нема познат структурен образец за
# реконструкција на изгубени карактери.
_TEXT_WRAPPER_RE = re.compile(
    r"\\(?:overline|underline|mathbb|mathrm|mathit|mathcal|boldsymbol|text|hat|vec|bar)"
    r"\s*\{([^{}]*)\}"
)

# \begin{matrix}...\end{matrix} (и pmatrix/bmatrix/array итн.) - моделот
# понекогаш "гледа" вишок ред (пр. линија од хартијата, сенка) и го враќа
# изразот завиткан во multi-row конструкција. За основна алгебра земаме
# само единствениот непразен ред.
_ENV_RE = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}(?:\{[^{}]*\})?")


def _reconstruct_matrix_row_as_subtraction(row: str) -> str:
    """Оваа апликација НИКОГАШ не поддржува матрици/linear algebra - секој
    \\begin{matrix} излез од моделот е гарантирано халуцинација, никогаш
    намерен внес. Емпириски утврден повторлив образец (3+ реални случаи):
    моделот го "гледа" минус знакот (-) како колонски разделувач (&)
    наместо оператор - пр. "4-3" -> матрица со колони "4" и "3", или
    "10-4" -> колони "10" и "-4". Го реконструираме минусот автоматски
    наместо да бараме рачна поправка секој пат."""
    if "&" not in row:
        return row
    cells = [c.strip() for c in row.split("&") if c.strip()]
    if not cells:
        return row
    result = cells[0]
    for cell in cells[1:]:
        result += cell if cell[0] in "+-" else f"-{cell}"
    return result


_BARE_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def _resolve_fraction_hallucination(numerator: str, denominator: str) -> str | None:
    """Друг повторлив, документиран образец на дропка-халуцинација (5+
    реални случаи): моделот "дуплира" еден операнд како именител/броител
    на измислена дропка - "5+5" -> \\frac{5+5}{5}, "1+1" -> \\frac{1}{1+1},
    "10+4" -> \\frac{10+4}{4}, "1+2" -> \\frac{1+2}{2}. Секогаш едната
    страна е гол број кој веќе се јавува како операнд во другата страна
    (која содржи оператор).

    Строго ограничено на ЧИСТА аритметика (без букви/променливи) за да
    не ги допре легитимните алгебарски дропки (пр. "(x+2)/2" НЕ се
    допира бидејќи "x" е буква).

    Враќа ја "вистинската" страна самостојно ако е препознаен образецот,
    инаку None (третирај го како вистинска дропка)."""
    if re.search(r"[a-zA-Z]", numerator) or re.search(r"[a-zA-Z]", denominator):
        return None

    num_bare = bool(_BARE_NUMBER_RE.match(numerator.strip()))
    den_bare = bool(_BARE_NUMBER_RE.match(denominator.strip()))

    if den_bare and not num_bare:
        operands = re.findall(r"\d+(?:\.\d+)?", numerator)
        if denominator.strip().lstrip("-") in operands:
            return numerator
    elif num_bare and not den_bare:
        operands = re.findall(r"\d+(?:\.\d+)?", denominator)
        if numerator.strip().lstrip("-") in operands:
            return denominator

    return None


def _strip_matrix_rows(text: str) -> str:
    text = _ENV_RE.sub("", text)
    if "\\\\" in text:
        rows = [r.strip() for r in text.split("\\\\")]
        rows = [r for r in rows if r]
        if rows:
            text = rows[0]
    text = _reconstruct_matrix_row_as_subtraction(text)
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


def _read_group(text: str, i: int):
    """Ако на позиција i (по празни места) почнува "{...}", враќа
    (содржина, позиција-по-затворената-заграда) со правилно броење на
    вгнездени загради, инаку None."""
    while i < len(text) and text[i] == " ":
        i += 1
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
    return None  # незатворена заграда


def _convert_frac_sqrt(text: str) -> str:
    out = []
    i = 0
    while i < len(text):
        if text.startswith("\\frac", i):
            first = _read_group(text, i + 5)
            second = _read_group(text, first[1]) if first else None
            if first and second:
                num = _convert_frac_sqrt(first[0])
                den = _convert_frac_sqrt(second[0])
                resolved = _resolve_fraction_hallucination(num, den)
                out.append(f"({resolved})" if resolved is not None else f"({num})/({den})")
                i = second[1]
                continue
        elif text.startswith("\\sqrt", i):
            group = _read_group(text, i + 5)
            if group:
                out.append(f"sqrt({_convert_frac_sqrt(group[0])})")
                i = group[1]
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def _convert_integral(text: str) -> str:
    """\\int f dx -> integrate(f,x). Модели го препознаваат неопределениот
    интеграл од слика (пр. \\int \\frac{dx}{\\sqrt{...}}), а нашиот парсер
    очекува integrate(израз,x). Само неопределени интеграли (без граници)."""
    if not text.lstrip().startswith("\\int"):
        return text
    body = re.sub(r"\s+", "", text.lstrip()[4:])

    # \int \frac{dx}{g}  ->  (dx)/(g)  =>  integrate(1/(g), x)
    m = re.fullmatch(r"\(d([a-z])\)/\((.+)\)", body)
    if m:
        return f"integrate(1/({m.group(2)}),{m.group(1)})"

    # \int f dx  =>  integrate(f, x)
    m = re.fullmatch(r"(.+?)d([a-z])", body)
    if m:
        return f"integrate({m.group(1)},{m.group(2)})"
    return text


def latex_to_plain(latex: str) -> str:
    """Претвора (основен подмножество) LaTeX во ASCII израз што
    normalize_text()/parse_expr() можат да го разберат."""
    text = latex.strip()
    text = _strip_matrix_rows(text)
    prev = None
    while prev != text:
        prev = text
        text = _TEXT_WRAPPER_RE.sub(r"\1", text)

    # \frac{a}{b} -> (a)/(b) и \sqrt{a} -> sqrt(a), со поддршка за
    # вгнездување (пр. \frac{dx}{\sqrt{1-x}}) - броиме загради наместо regex.
    text = _convert_frac_sqrt(text)

    # Напишан знак "^" (пр. "x^2" како буквален текст) моделот го чита како
    # LaTeX \wedge (симболот ∧ изгледа исто) - или "^{\wedge}2", или
    # "\wedge 2". Реален случај: "diff(x^2,x)" -> "diff(x^{\wedge}2,x)".
    text = re.sub(r"\^\s*\{\s*\\wedge\s*\}", "^", text)
    text = text.replace("\\wedge", "^")

    # Истиот проблем, друга форма: буквален "^" се чита како подигната "1" -
    # "x^3" -> "x^{1}3". Степен 1 директно пред цифра е бесмислен ("x^1*3"
    # никој не го пишува така), па е сигурно "^" што го прочитал како 1.
    text = re.sub(r"\^\s*\{\s*1\s*\}(?=\d)", "^", text)

    text = _SQRT_RE.sub(r"sqrt(\1)", text)
    text = _BRACED_POWER_RE.sub(r"^(\1)", text)
    text = _SUBSCRIPT_RE.sub("", text)

    for bad, good in _LATEX_REPLACEMENTS.items():
        text = text.replace(bad, good)

    text = text.replace("{", "(").replace("}", ")")

    text = _convert_integral(text)

    # Safety net: секој преостанат "\" (пр. непозната LaTeX команда што
    # ја нема во _LATEX_REPLACEMENTS, или "\_" остаток по SUBSCRIPT_RE)
    # никогаш не е валиден во финалниот ASCII израз - отстрани го самиот
    # backslash (не и текстот околу него, пр. "\alpha" -> "alpha").
    text = text.replace("\\", "")

    return text


@dataclass
class ParsedProblem:
    raw_text: str
    normalized_text: str
    kind: str  # "equation" | "expression" | "derivative" | "integral" | "system"
    sympy_obj: object  # Eq(...) / Expr / list[Eq] (за "system")
    variables: list = field(default_factory=list)
    op_var: object = None  # Symbol - само за "derivative"/"integral" (по која променлива)


def normalize_text(raw: str) -> str:
    """Чисти суров текст (од OCR или рачен внес) во форма што SymPy
    парсерот може да ја обработи."""
    text = raw.strip()

    for bad, good in _SYMBOL_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = _DECIMAL_COMMA_RE.sub(".", text)

    # Отстрани празни места - "2 x + 3" -> "2x+3"
    text = "".join(text.split())

    # Израз никогаш не почнува со "*" - остаток од \cdot/\times што моделот
    # го "измислил" пред изразот (реален случај: "**diff(x^2,x)").
    text = text.lstrip("*")

    # "1" (цифра) визуелно наликува на "i"/"l" (мало L) - чест OCR/
    # ракописен misread, реален случај забележан во UI ("1+1" -> "i+1").
    # Замени ги САМО кога не се дел од подолг збор (функција/променлива,
    # пр. "sin", "diff", "integrate") - таму секое "i"/"l" е опкружено со
    # други букви, па оваа замена никогаш не ги допира.
    text = re.sub(r"(?<![a-zA-Z])[il](?![a-zA-Z])", "1", text)

    # sqrt без загради, пр. "sqrt9" -> "sqrt(9)" - чест случај кога OCR
    # го изгуби заградата или корисникот ја испуштил
    text = re.sub(r"sqrt(\d+(\.\d+)?)", r"sqrt(\1)", text)
    text = re.sub(r"sqrt([a-zA-Z])(?!\()", r"sqrt(\1)", text)

    # sin/cos/tan без загради, пр. "sin30" -> "sin(30)" - без ова SymPy
    # ги дели буквите s,i,n како посебни променливи (implicit multiplication)
    # наместо да ја препознае функцијата, реален случај забележан во UI.
    for _fn in ("sin", "cos", "tan"):
        text = re.sub(rf"{_fn}(\d+(\.\d+)?)", rf"{_fn}(\1)", text)
        text = re.sub(rf"{_fn}([a-zA-Z])(?!\()", rf"{_fn}(\1)", text)

    # Отстрани trailing "..." - забележан артефакт кај handwriting моделот
    # (генеративен модел понекогаш "продолжува" со точки на крајот наместо
    # чисто да застане). Еден единствен trailing "." (децимала без остаток,
    # пр. "5.") НЕ се допира - ретко и не вреди ризикот.
    text = re.sub(r"\.{2,}$", "", text)

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
        bad_chars = sorted(set(re.sub(r"[0-9a-zA-Z+\-*/^.,=()<>;]", "", text)))
        raise ParseError(
            f"Изразот содржи непрепознаени карактери ({''.join(bad_chars)}) - "
            f"веројатно грешка во препознавањето. Провери/поправи го текстот."
        )

    return text


_TRIG_FUNCS = (sin, cos, tan)


def _degrees_to_radians_in_trig(expr):
    """sin(30) во школски контекст значи 30°, не 30 радијани (SymPy
    default). Конвертираме sin/cos/tan САМО кога аргументот е чист број
    БЕЗ pi во него (пр. sin(30) -> sin(30°), но sin(pi/6) си останува
    радијани - експлицитноे бара корисникот со pi). Симболични аргументи
    (пр. sin(x) во извод/интеграл) НЕ се допираат - таму мора радијани
    за да важат стандардните правила за изводи/интеграли."""
    replacements = {}
    for func in _TRIG_FUNCS:
        for node in expr.atoms(func):
            arg = node.args[0]
            if arg.is_number and not arg.has(pi):
                replacements[node] = func(arg * pi / 180)
    return expr.xreplace(replacements) if replacements else expr


def _safe_parse_expr(text: str):
    """parse_expr() со единствено, конзистентно фаќање грешки - секое
    "смет" (garbage) влезно парче (од OCR или рачен внес) станува чист
    ParseError, никогаш необработен crash."""
    try:
        expr = parse_expr(text, transformations=_TRANSFORMATIONS)
        return _degrees_to_radians_in_trig(expr)
    except Exception as exc:
        raise ParseError(
            f"Не можам да го парсирам изразот '{text}'. "
            f"Провери дали е точно препознаен/внесен."
        ) from exc


def _parse_equation(raw_text: str, normalized: str) -> ParsedProblem:
    if normalized.count("=") > 1:
        raise ParseError("Изразот содржи повеќе од еден знак '=' - не можам да го решам.")

    lhs_str, rhs_str = normalized.split("=")
    lhs = _safe_parse_expr(lhs_str)
    rhs = _safe_parse_expr(rhs_str)
    sympy_obj = Eq(lhs, rhs)

    return ParsedProblem(
        raw_text=raw_text,
        normalized_text=normalized,
        kind="equation",
        sympy_obj=sympy_obj,
        variables=sorted(sympy_obj.free_symbols, key=lambda s: s.name),
    )


def _parse_expression(raw_text: str, normalized: str) -> ParsedProblem:
    sympy_obj = _safe_parse_expr(normalized)
    return ParsedProblem(
        raw_text=raw_text,
        normalized_text=normalized,
        kind="expression",
        sympy_obj=sympy_obj,
        variables=sorted(sympy_obj.free_symbols, key=lambda s: s.name),
    )


def _parse_calculus(raw_text: str, normalized: str, match: re.Match, kind: str) -> ParsedProblem:
    inner_str, var_str = match.group(1), match.group(2)
    inner_expr = _safe_parse_expr(inner_str)
    op_var = Symbol(var_str)

    return ParsedProblem(
        raw_text=raw_text,
        normalized_text=normalized,
        kind=kind,
        sympy_obj=inner_expr,
        variables=sorted(inner_expr.free_symbols, key=lambda s: s.name),
        op_var=op_var,
    )


def _parse_system(raw_text: str, normalized: str) -> ParsedProblem:
    parts = [p for p in normalized.split(";") if p]
    if len(parts) < 2:
        raise ParseError(
            "Системот равенки треба да содржи најмалку 2 равенки, "
            "одделени со ';' (пр. 'x+y=5;x-y=1')."
        )

    equations = []
    all_vars = set()
    for part in parts:
        if part.count("=") != 1:
            raise ParseError(f"'{part}' не е валидна равенка (треба точно еден знак '=').")
        lhs_str, rhs_str = part.split("=")
        lhs = _safe_parse_expr(lhs_str)
        rhs = _safe_parse_expr(rhs_str)
        eq = Eq(lhs, rhs)
        equations.append(eq)
        all_vars |= eq.free_symbols

    return ParsedProblem(
        raw_text=raw_text,
        normalized_text=normalized,
        kind="system",
        sympy_obj=equations,
        variables=sorted(all_vars, key=lambda s: s.name),
    )


def parse(raw_text: str) -> ParsedProblem:
    """
    Главна функција: суров текст -> ParsedProblem со sympy објект.

    Поддржува равенки ("2x+3=11"), обични изрази за симплификација
    ("2x+3(x-1)"), деривати ("diff(x^2+3x,x)"), интеграли
    ("integrate(x^2,x)") и системи равенки ("x+y=5;x-y=1").
    """
    normalized = normalize_text(raw_text)

    if ";" in normalized:
        return _parse_system(raw_text, normalized)

    diff_match = _DIFF_CALL_RE.match(normalized)
    if diff_match:
        return _parse_calculus(raw_text, normalized, diff_match, kind="derivative")

    integrate_match = _INTEGRATE_CALL_RE.match(normalized)
    if integrate_match:
        return _parse_calculus(raw_text, normalized, integrate_match, kind="integral")

    if "=" in normalized:
        return _parse_equation(raw_text, normalized)

    return _parse_expression(raw_text, normalized)


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
        op_var=parsed.op_var,
    )
