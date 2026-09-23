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


_SYMBOL_REPLACEMENTS = {
    "×": "*",
    "·": "*",
    "÷": "/",
    "−": "-",
    "—": "-",
    "–": "-",
    "√": "sqrt",
    "π": "pi",
    "²": "**2",
    "³": "**3",
}

_DECIMAL_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")


class ParseError(ValueError):
    pass


_ALLOWED_CHARS_RE = re.compile(r"[0-9a-zA-Z+\-*/^.,=()<>;]*")


_DIFF_CALL_RE = re.compile(r"^diff\((.+),([a-zA-Z]\w*)\)$")
_INTEGRATE_CALL_RE = re.compile(r"^integrate\((.+),([a-zA-Z]\w*)\)$")


_FRAC_RE = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\s*\{([^{}]*)\}")
_BRACED_POWER_RE = re.compile(r"\^\s*\{([^{}]*)\}")
_SUBSCRIPT_RE = re.compile(r"_\s*\{?[^{}\s]*\}?")


_TEXT_WRAPPER_RE = re.compile(
    r"\\(?:overline|underline|mathbb|mathrm|mathit|mathcal|boldsymbol|text|hat|vec|bar)"
    r"\s*\{([^{}]*)\}"
)


_ENV_RE = re.compile(r"\\(?:begin|end)\{[a-zA-Z*]+\}(?:\{[^{}]*\})?")


def _reconstruct_matrix_row_as_subtraction(row: str) -> str:

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
    return None


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

    if not text.lstrip().startswith("\\int"):
        return text
    body = re.sub(r"\s+", "", text.lstrip()[4:])

    m = re.fullmatch(r"\(d([a-z])\)/\((.+)\)", body)
    if m:
        return f"integrate(1/({m.group(2)}),{m.group(1)})"

    m = re.fullmatch(r"(.+?)d([a-z])", body)
    if m:
        return f"integrate({m.group(1)},{m.group(2)})"
    return text


def latex_to_plain(latex: str) -> str:

    text = latex.strip()
    text = _strip_matrix_rows(text)
    prev = None
    while prev != text:
        prev = text
        text = _TEXT_WRAPPER_RE.sub(r"\1", text)

    text = _convert_frac_sqrt(text)

    text = re.sub(r"\^\s*\{\s*\\wedge\s*\}", "^", text)
    text = text.replace("\\wedge", "^")
    text = re.sub(r"\^\s*\{\s*1\s*\}(?=\d)", "^", text)

    text = _SQRT_RE.sub(r"sqrt(\1)", text)
    text = _BRACED_POWER_RE.sub(r"^(\1)", text)
    text = _SUBSCRIPT_RE.sub("", text)

    for bad, good in _LATEX_REPLACEMENTS.items():
        text = text.replace(bad, good)

    text = text.replace("{", "(").replace("}", ")")

    text = _convert_integral(text)

    text = text.replace("\\", "")

    return text


@dataclass
class ParsedProblem:
    raw_text: str
    normalized_text: str
    kind: str
    sympy_obj: object
    variables: list = field(default_factory=list)
    op_var: object = None


def normalize_text(raw: str) -> str:

    text = raw.strip()

    for bad, good in _SYMBOL_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = _DECIMAL_COMMA_RE.sub(".", text)

    text = "".join(text.split())

    text = text.lstrip("*")

    text = re.sub(r"(?<![a-zA-Z])[il](?![a-zA-Z])", "1", text)

    text = re.sub(r"sqrt(\d+(\.\d+)?)", r"sqrt(\1)", text)
    text = re.sub(r"sqrt([a-zA-Z])(?!\()", r"sqrt(\1)", text)

    for _fn in ("sin", "cos", "tan"):
        text = re.sub(rf"{_fn}(\d+(\.\d+)?)", rf"{_fn}(\1)", text)
        text = re.sub(rf"{_fn}([a-zA-Z])(?!\()", rf"{_fn}(\1)", text)

    text = re.sub(r"\.{2,}$", "", text)

    if not text:
        raise ParseError("Празен израз - нема што да се препознае/парсира.")

    if not _ALLOWED_CHARS_RE.fullmatch(text):
        bad_chars = sorted(set(re.sub(r"[0-9a-zA-Z+\-*/^.,=()<>;]", "", text)))
        raise ParseError(
            f"Изразот содржи непрепознаени карактери ({''.join(bad_chars)}) - "
            f"веројатно грешка во препознавањето. Провери/поправи го текстот."
        )

    return text


_TRIG_FUNCS = (sin, cos, tan)


def _degrees_to_radians_in_trig(expr):

    replacements = {}
    for func in _TRIG_FUNCS:
        for node in expr.atoms(func):
            arg = node.args[0]
            if arg.is_number and not arg.has(pi):
                replacements[node] = func(arg * pi / 180)
    return expr.xreplace(replacements) if replacements else expr


def _safe_parse_expr(text: str):

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
