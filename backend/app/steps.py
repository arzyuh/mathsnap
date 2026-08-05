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


def solve_polynomial_factoring(eq: Eq, var: Symbol) -> Method | None:
    """Обидува се да ја реши со факторизација - работи за КОЈ БИЛО степен
    (квадратна, кубна, квартична...), не само квадратни. Враќа None ако
    изразот не се факторизира убаво над рационални броеви (тогаш нема
    смисла да се нуди овој метод на корисникот)."""
    moved = expand(eq.lhs - eq.rhs)

    factored = factor(moved)
    if factored == moved or not factored.is_Mul:
        return None  # не се факторизира убаво - прескокни го методот

    method = Method(name="Факторизација")
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


def solve_polynomial_general_roots(eq: Eq, var: Symbol, degree: int) -> Method:
    """Fallback за равенки со степен >=3 кои не се факторизираат убаво -
    чекор-по-чекор извод на формула за кубни/квартични равенки е вон
    опфатот на овој MVP (многу комплексно), затоа ги прикажуваме сите
    корени (реални и комплексни) директно преку SymPy solve()."""
    moved = expand(eq.lhs - eq.rhs)

    method = Method(name=f"Општо решение (степен {degree})")
    method.add("Почетна равенка", eq)
    method.add("Пренеси ги сите членови на левата страна", Eq(moved, 0))
    method.add(
        f"Равенка од {degree}-ти степен без убава факторизација - "
        f"решена директно (формулите за степен ≥3 се многу комплексни за чекор-по-чекор приказ)",
        Eq(moved, 0),
    )

    roots = sympy.solve(Eq(moved, 0), var)
    if not roots:
        method.result_text = "Нема решение во затворена форма"
        method.result_latex = r"\text{нема решение}"
        return method

    method.add("Корени", sympy.FiniteSet(*roots))
    method.result_text = ", ".join(f"x{i+1} = {r}" for i, r in enumerate(roots))
    method.result_latex = ", \\quad ".join(f"x_{i+1} = {_lx(r)}" for i, r in enumerate(roots))
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


# ---------------------------------------------------------------------------
# Деривати: diff(израз, променлива)
# ---------------------------------------------------------------------------

_KNOWN_DERIVATIVES = {
    sympy.sin: (lambda a: sympy.cos(a), "d/dx[sin(x)] = cos(x)"),
    sympy.cos: (lambda a: -sympy.sin(a), "d/dx[cos(x)] = -sin(x)"),
    sympy.tan: (lambda a: 1 + sympy.tan(a) ** 2, "d/dx[tan(x)] = 1+tan²(x)"),
    sympy.exp: (lambda a: sympy.exp(a), "d/dx[e^x] = e^x"),
    sympy.log: (lambda a: 1 / a, "d/dx[ln(x)] = 1/x"),
    sqrt: (lambda a: 1 / (2 * sqrt(a)), "d/dx[√x] = 1/(2√x)"),
}


def _differentiate_term(term, var: Symbol):
    """Врaќа (име_на_правило, изведен_член). Ги покрива честите
    учебнички случаи (степен, познати функции); за сè посложено
    (chain rule) паѓа на директен sympy.diff (сепак точно, само без
    детален меѓучекор)."""
    if var not in term.free_symbols:
        return "Изводот на константа е 0", sympy.Integer(0)

    if term == var:
        return "Изводот на x е 1", sympy.Integer(1)

    coeff, rest = term.as_coeff_Mul()

    if rest == var:
        return f"Константен множител: d/dx[{coeff}·x] = {coeff}", coeff

    if rest.is_Pow and rest.base == var and not rest.exp.has(var):
        n = rest.exp
        deriv = simplify(coeff * n * var ** (n - 1))
        return f"Правило на степен: d/dx[x^{n}] = {n}·x^({n}-1)", deriv

    if rest.func in _KNOWN_DERIVATIVES and rest.args and rest.args[0] == var:
        deriv_fn, label = _KNOWN_DERIVATIVES[rest.func]
        return label, simplify(coeff * deriv_fn(var))

    return "Примени го правилото за синџир/изводи", sympy.diff(term, var)


def differentiate_steps(expr, var: Symbol) -> Method:
    method = Method(name="Извод (диференцирање)")
    method.add(f"Најди го изводот по {var}", sympy.Derivative(expr, var, evaluate=False))

    terms = sympy.Add.make_args(expand(expr))
    if len(terms) > 1:
        method.add(
            "Правило за збир - изведи го секој член поединечно",
            sympy.Add(*[sympy.Derivative(t, var, evaluate=False) for t in terms]),
        )

    for term in terms:
        rule_name, term_deriv = _differentiate_term(term, var)
        method.add(rule_name, Eq(sympy.Derivative(term, var, evaluate=False), term_deriv))

    result = sympy.diff(expr, var)  # secutiy net - секогаш точен вкупен резултат
    method.add("Собери ги сите изведени членови", result)
    method.result_text = str(result)
    method.result_latex = _lx(result)
    return method


# ---------------------------------------------------------------------------
# Интеграли: integrate(израз, променлива)
# ---------------------------------------------------------------------------

_KNOWN_INTEGRALS = {
    sympy.sin: (lambda a: -sympy.cos(a), "∫sin(x) dx = -cos(x)"),
    sympy.cos: (lambda a: sympy.sin(a), "∫cos(x) dx = sin(x)"),
    sympy.exp: (lambda a: sympy.exp(a), "∫e^x dx = e^x"),
}


def _integrate_term(term, var: Symbol):
    if var not in term.free_symbols:
        return f"Интеграл на константа: ∫{term} dx = {term}·x", term * var

    coeff, rest = term.as_coeff_Mul()

    if rest == var:
        return "Правило на степен (n=1): ∫x dx = x²/2", coeff * var ** 2 / 2

    if rest.is_Pow and rest.base == var and not rest.exp.has(var):
        n = rest.exp
        if n == -1:
            return "∫(1/x) dx = ln|x|", coeff * sympy.log(sympy.Abs(var))
        new_n = n + 1
        integral = simplify(coeff * var ** new_n / new_n)
        return f"Правило на степен: ∫x^{n} dx = x^({n}+1)/({n}+1)", integral

    if rest.func in _KNOWN_INTEGRALS and rest.args and rest.args[0] == var:
        int_fn, label = _KNOWN_INTEGRALS[rest.func]
        return label, simplify(coeff * int_fn(var))

    return "Општо решение (посложен член)", sympy.integrate(term, var)


def integrate_steps(expr, var: Symbol) -> Method:
    method = Method(name="Интеграл (антидеривација)")
    method.add(f"Најди ја антидеривативата по {var}", sympy.Integral(expr, var))

    terms = sympy.Add.make_args(expand(expr))
    if len(terms) > 1:
        method.add(
            "Правило за збир - интегрирај го секој член поединечно",
            sympy.Add(*[sympy.Integral(t, var) for t in terms]),
        )

    for term in terms:
        rule_name, term_int = _integrate_term(term, var)
        method.add(rule_name, Eq(sympy.Integral(term, var), term_int))

    result = sympy.integrate(expr, var)  # security net - секогаш точен резултат
    final = result + Symbol("C")
    method.add("Собери ги сите членови (+ константа на интеграција C)", final)
    method.result_text = str(final)
    method.result_latex = _lx(final)
    return method


# ---------------------------------------------------------------------------
# Системи од 2 линеарни равенки со 2 непознати (метод на замена)
# ---------------------------------------------------------------------------

def solve_linear_system_2x2(eq1: Eq, eq2: Eq, var1: Symbol, var2: Symbol) -> Method:
    method = Method(name="Систем равенки - метод на замена")
    method.add("Прва равенка", eq1)
    method.add("Втора равенка", eq2)

    # Изрази var1 преку var2 од првата равенка
    expr_var1 = sympy.solve(eq1, var1)
    if not expr_var1:
        # var1 го нема во првата равенка - пробај со втората
        eq1, eq2 = eq2, eq1
        expr_var1 = sympy.solve(eq1, var1)
    expr_var1 = expr_var1[0]

    method.add(f"Изрази {var1} преку {var2} од првата равенка", Eq(var1, expr_var1))

    substituted = eq2.subs(var1, expr_var1)
    method.add(f"Замени во втората равенка", substituted)

    var2_solutions = sympy.solve(substituted, var2)
    if not var2_solutions:
        method.result_text = "Нема единствено решение (системот е противречен или зависен)"
        method.result_latex = r"\text{нема единствено решение}"
        return method

    var2_value = var2_solutions[0]
    method.add(f"Реши по {var2}", Eq(var2, var2_value))

    var1_value = simplify(expr_var1.subs(var2, var2_value))
    method.add(f"Замени {var2}={var2_value} назад за да го најдеш {var1}", Eq(var1, var1_value))

    method.result_text = f"{var1} = {var1_value}, {var2} = {var2_value}"
    method.result_latex = f"{_lx(var1)} = {_lx(var1_value)}, \\quad {_lx(var2)} = {_lx(var2_value)}"
    return method


# ---------------------------------------------------------------------------
# Основни тригонометриски равенки: sin(x)=a, cos(x)=a, tan(x)=a
# ---------------------------------------------------------------------------

def solve_basic_trig_equation(eq: Eq, var: Symbol) -> Method | None:
    """Препознава равенки од обликот sin(x)=a / cos(x)=a / tan(x)=a
    (по преместување на сите членови). Враќа None ако равенката не е
    од овој едноставен облик (тогаш го користи општото solve()-падобран)."""
    moved = expand(eq.lhs - eq.rhs)
    trig_terms = [t for t in moved.atoms(sympy.sin, sympy.cos, sympy.tan) if var in t.free_symbols]
    if len(trig_terms) != 1:
        return None

    trig_term = trig_terms[0]
    if trig_term.args[0] != var:
        return None  # посложен аргумент (пр. sin(2x)) - вон опфат на MVP

    rest = simplify(moved - trig_term)  # moved = trig_term + rest = 0  =>  trig_term = -rest
    rhs_value = simplify(-rest)

    method = Method(name="Тригонометриска равенка")
    method.add("Почетна равенка", eq)
    method.add(f"Изолирај ја тригонометриската функција", Eq(trig_term, rhs_value))

    n = Symbol("n", integer=True)
    func = trig_term.func

    if abs(rhs_value) > 1 and func in (sympy.sin, sympy.cos):
        method.result_text = "Нема реални решенија (|вредност| > 1)"
        method.result_latex = r"\text{нема реални решенија}"
        return method

    if func == sympy.sin:
        base = sympy.asin(rhs_value)
        method.add("Примени arcsin на двете страни", Eq(var, base))
        general = sympy.Or(
            Eq(var, base + 2 * sympy.pi * n),
            Eq(var, sympy.pi - base + 2 * sympy.pi * n),
        )
        method.result_text = f"x = {base} + 2πn  или  x = π - ({base}) + 2πn,  n ∈ ℤ"
    elif func == sympy.cos:
        base = sympy.acos(rhs_value)
        method.add("Примени arccos на двете страни", Eq(var, base))
        general = Eq(var, sympy.Or(base, -base) + 2 * sympy.pi * n)
        method.result_text = f"x = ±{base} + 2πn,  n ∈ ℤ"
    else:  # tan
        base = sympy.atan(rhs_value)
        method.add("Примени arctan на двете страни", Eq(var, base))
        general = Eq(var, base + sympy.pi * n)
        method.result_text = f"x = {base} + πn,  n ∈ ℤ"

    method.add("Општо решение (периодично)", general)
    method.result_latex = _lx(general)
    return method
