from __future__ import annotations

import sympy
from sympy import Poly, expand

from .expression_parser import ParsedProblem
from .steps import (
    Method,
    differentiate_steps,
    integrate_steps,
    simplify_expression_steps,
    solve_basic_trig_equation,
    solve_linear_equation,
    solve_linear_system_2x2,
    solve_polynomial_factoring,
    solve_polynomial_general_roots,
    solve_quadratic_formula,
)


class UnsupportedProblem(Exception):
    pass


def _fallback_generic_solve(parsed: ParsedProblem) -> Method:

    method = Method(name="Општо решение (SymPy)")
    if parsed.kind == "equation":
        var = parsed.variables[0] if parsed.variables else None
        if var is None:
            raise UnsupportedProblem("Равенката нема непозната променлива.")
        solutions = sympy.solve(parsed.sympy_obj, var)
        method.add("Почетна равенка", parsed.sympy_obj)
        method.result_text = ", ".join(f"x = {s}" for s in solutions) or "нема решение"
        method.result_latex = ", ".join(f"x = {sympy.latex(s)}" for s in solutions)
    else:
        simplified = sympy.simplify(parsed.sympy_obj)
        method.add("Почетен израз", parsed.sympy_obj)
        method.result_text = str(simplified)
        method.result_latex = sympy.latex(simplified)
    return method


def _solve_system(parsed: ParsedProblem) -> dict:
    result = {
        "kind": "system",
        "problem_type": "system",
        "variable": None,
        "methods": [],
        "graphable": False,
    }
    equations = parsed.sympy_obj
    variables = parsed.variables

    if len(equations) == 2 and len(variables) == 2:
        var1, var2 = variables
        result["methods"] = [solve_linear_system_2x2(equations[0], equations[1], var1, var2)]
        return result

    method = Method(name="Општо решение на систем (SymPy)")
    for i, eq in enumerate(equations):
        method.add(f"Равенка {i + 1}", eq)
    solutions = sympy.solve(equations, variables)
    if solutions:
        if isinstance(solutions, dict):
            pairs = [f"{k} = {v}" for k, v in solutions.items()]
        else:
            pairs = [str(solutions)]
        method.result_text = ", ".join(pairs)
        method.result_latex = ", \\quad ".join(sympy.latex(sympy.Eq(k, v)) for k, v in solutions.items()) if isinstance(solutions, dict) else str(solutions)
    else:
        method.result_text = "Нема решение"
        method.result_latex = r"\text{нема решение}"
    result["methods"] = [method]
    return result


def solve(parsed: ParsedProblem) -> dict:

    result = {
        "kind": parsed.kind,
        "problem_type": "general",
        "variable": None,
        "methods": [],
        "graphable": False,
    }

    if parsed.kind == "expression":
        result["problem_type"] = "simplify"
        result["methods"] = [simplify_expression_steps(parsed.sympy_obj)]
        return result

    if parsed.kind == "derivative":
        result["problem_type"] = "derivative"
        result["variable"] = str(parsed.op_var)
        result["methods"] = [differentiate_steps(parsed.sympy_obj, parsed.op_var)]
        return result

    if parsed.kind == "integral":
        result["problem_type"] = "integral"
        result["variable"] = str(parsed.op_var)
        result["methods"] = [integrate_steps(parsed.sympy_obj, parsed.op_var)]
        return result

    if parsed.kind == "system":
        return _solve_system(parsed)

    if len(parsed.variables) != 1:

        result["methods"] = [_fallback_generic_solve(parsed)]
        return result

    var = parsed.variables[0]
    result["variable"] = str(var)

    trig_method = solve_basic_trig_equation(parsed.sympy_obj, var)
    if trig_method is not None:
        result["problem_type"] = "trigonometric"
        result["methods"] = [trig_method]
        return result

    try:
        degree = Poly(expand(parsed.sympy_obj.lhs - parsed.sympy_obj.rhs), var).degree()
    except sympy.PolynomialError:
        result["methods"] = [_fallback_generic_solve(parsed)]
        return result

    if degree == 1:
        result["problem_type"] = "linear"
        result["methods"] = [solve_linear_equation(parsed.sympy_obj, var)]
    elif degree == 2:
        result["problem_type"] = "quadratic"
        methods = []
        factoring = solve_polynomial_factoring(parsed.sympy_obj, var)
        if factoring is not None:
            methods.append(factoring)
        methods.append(solve_quadratic_formula(parsed.sympy_obj, var))
        result["methods"] = methods
    elif degree >= 3:
        result["problem_type"] = "higher_degree"
        factoring = solve_polynomial_factoring(parsed.sympy_obj, var)
        if factoring is not None:
            result["methods"] = [factoring]
        else:
            result["methods"] = [solve_polynomial_general_roots(parsed.sympy_obj, var, degree)]
    else:
        result["methods"] = [_fallback_generic_solve(parsed)]

    return result


def method_to_dict(method: Method) -> dict:
    return {
        "name": method.name,
        "steps": [
            {"explanation": s.explanation, "expr_latex": s.expr_latex}
            for s in method.steps
        ],
        "result_text": method.result_text,
        "result_latex": method.result_latex,
    }
