"""
Rule-based "narrator" за чекор-по-чекор решавање.

SymPy знае да го реши изразот, но не објаснува КАКО. Овој модул го
става тој наратив: секој метод произведува листа од Step-ови
(објаснување + меѓу-резултат), исто како што Photomath прикажува
"1. Растави ги заградите, 2. Пренеси членови..." итн.

Опфат (согласно договорениот MVP scope): линеарни равенки, квадратни
равенки (2 методи: факторизација + формула) и симплификација на изрази.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sympy
from sympy import Eq, Poly, Symbol, factor, expand, simplify, sqrt, latex


@dataclass
class Step:
    explanation: str
    expr_latex: str


@dataclass
class Method:
    name: str
    steps: list = field(default_factory=list)
    result_latex: str = ""
    result_text: str = ""

    def add(self, explanation: str, sympy_expr) -> None:
        self.steps.append(Step(explanation, latex(sympy_expr)))


def _lx(expr) -> str:
    return latex(expr)


# ---------------------------------------------------------------------------
# Линеарни равенки: ax + b = 0  (degree 1)
# ---------------------------------------------------------------------------

def solve_linear_equation(eq: Eq, var: Symbol) -> Method:
    method = Method(name="Линеарна равенка - изолирање на непознатата")

    lhs, rhs = eq.lhs, eq.rhs
    method.add("Почетна равенка", eq)

    expanded_lhs, expanded_rhs = expand(lhs), expand(rhs)
    if expanded_lhs != lhs or expanded_rhs != rhs:
        method.add("Раствори ги заградите", Eq(expanded_lhs, expanded_rhs))

    moved = expand(expanded_lhs - expanded_rhs)
    poly = Poly(moved, var)
    coeffs = poly.all_coeffs()  # [a, b] подредени од највисок кон најнизок степен
    while len(coeffs) < 2:
        coeffs.insert(0, 0)  # пополни ги повисоките (отсутни) степени со 0
    a, b = coeffs

    method.add(
        "Пренеси ги сите членови на левата страна (одземи ја десната страна од двете страни)",
        Eq(a * var + b, 0),
    )

    if a == 0:
        if b == 0:
            method.result_text = "Бесконечно многу решенија (важи за секој x)"
            method.result_latex = r"x \in \mathbb{R}"
        else:
            method.result_text = "Нема решение"
            method.result_latex = r"\text{нема решение}"
        return method

    if b != 0:
        method.add(f"Пренеси го слободниот член на десната страна", Eq(a * var, -b))

    solution = simplify(-b / a)
    if a != 1:
        method.add(f"Подели ги двете страни со коефициентот пред {var}", Eq(var, solution))
    else:
        method.add(f"{var} е веќе изолиран", Eq(var, solution))

    method.result_text = str(solution)
    method.result_latex = _lx(solution)
    return method


# ---------------------------------------------------------------------------
# Квадратни равенки: ax^2 + bx + c = 0  (degree 2)
# ---------------------------------------------------------------------------

def _quadratic_coeffs(eq: Eq, var: Symbol):
    moved = expand(eq.lhs - eq.rhs)
    poly = Poly(moved, var)
    coeffs = poly.all_coeffs()
    while len(coeffs) < 3:
        coeffs.insert(0, 0)
    a, b, c = coeffs
    return a, b, c, moved


def solve_quadratic_factoring(eq: Eq, var: Symbol) -> Method | None:
    """Обидува се да ја реши со факторизација. Враќа None ако изразот
    не се факторизира убаво над рационални броеви (тогаш нема смисла
    да се нуди овој метод на корисникот)."""
    a, b, c, moved = _quadratic_coeffs(eq, var)

    factored = factor(moved)
    if factored == moved or not factored.is_Mul:
        return None  # не се факторизира убаво - прескокни го методот

    method = Method(name="Квадратна равенка - факторизација")
    method.add("Почетна равенка", eq)
    method.add("Пренеси ги сите членови на левата страна", Eq(moved, 0))
    method.add("Факторизирај го изразот", Eq(factored, 0))

    roots = []
    for arg in factored.args:
        if var not in arg.free_symbols:
            continue
        sub_eq = Eq(arg, 0)
        method.add(f"Производот е нула ⇒ секој фактор поединечно е нула", sub_eq)
        sols = sympy.solve(sub_eq, var)
        roots.extend(sols)

    roots = sorted(set(roots), key=str)
    result_str = ", ".join(f"x = {r}" for r in roots)
    method.add("Решенија", Eq(var, roots[0]) if len(roots) == 1 else sympy.FiniteSet(*roots))
    method.result_text = result_str
    method.result_latex = ", ".join(f"x = {_lx(r)}" for r in roots)
    return method


def solve_quadratic_formula(eq: Eq, var: Symbol) -> Method:
    a, b, c, moved = _quadratic_coeffs(eq, var)

    method = Method(name="Квадратна равенка - квадратна формула")
    method.add("Почетна равенка", eq)
    method.add("Пренеси ги сите членови на левата страна", Eq(moved, 0))
    method.add(
        f"Идентификувај ги коефициентите: a = {a}, b = {b}, c = {c}",
        Eq(a * var**2 + b * var + c, 0),
    )

    discriminant = expand(b**2 - 4 * a * c)
    method.add(
        r"Пресметај ја дискриминантата: D = b^2 - 4ac",
        Eq(Symbol("D"), discriminant),
    )

    formula_lhs = Symbol("x")
    if discriminant < 0:
        real_part = simplify(-b / (2 * a))
        imag_part = simplify(sqrt(-discriminant) / (2 * a))
        method.add(
            "D < 0 ⇒ нема реални решенија, но постојат две комплексни решенија",
            sympy.Eq(
                formula_lhs,
                real_part + imag_part * sympy.I,
            ),
        )
        x1 = real_part + imag_part * sympy.I
        x2 = real_part - imag_part * sympy.I
        method.result_text = f"x1 = {x1}, x2 = {x2} (комплексни решенија)"
        method.result_latex = f"x_{{1,2}} = {_lx(real_part)} \\pm {_lx(imag_part)}i"
        return method

    sqrt_d = sqrt(discriminant)
    x1 = simplify((-b + sqrt_d) / (2 * a))
    x2 = simplify((-b - sqrt_d) / (2 * a))

    method.add(
        r"Примени ја квадратната формула: x = (-b \pm \sqrt{D}) / (2a)",
        Eq(formula_lhs, (-b + sympy.sqrt(Symbol("D"))) / (2 * a)),
    )

    if discriminant == 0:
        method.add("D = 0 ⇒ едно (двојно) решение", Eq(var, x1))
        method.result_text = f"x = {x1}"
        method.result_latex = f"x = {_lx(x1)}"
    else:
        method.add("D > 0 ⇒ две различни реални решенија", sympy.FiniteSet(x1, x2))
        method.result_text = f"x1 = {x1}, x2 = {x2}"
        method.result_latex = f"x_1 = {_lx(x1)}, \\quad x_2 = {_lx(x2)}"

    return method


# ---------------------------------------------------------------------------
# Симплификација на обични изрази (без "=")
# ---------------------------------------------------------------------------

def simplify_expression_steps(expr) -> Method:
    method = Method(name="Симплификација")
    method.add("Почетен израз", expr)

    expanded = expand(expr)
    if expanded != expr:
        method.add("Раствори ги заградите и собери слични членови", expanded)

    simplified = simplify(expr)
    if simplified != expanded:
        method.add("Дополнително поедностави (скрати, факторизирај)", simplified)

    method.result_text = str(simplified)
    method.result_latex = _lx(simplified)
    return method
