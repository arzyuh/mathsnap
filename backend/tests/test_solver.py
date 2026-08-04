"""
Основни тестови за parser + solver pipeline-от.
Стартување (од backend/ папката): python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.expression_parser import latex_to_plain, parse, parse_latex
from app.solver import solve


def test_linear_equation_simple():
    parsed = parse("2x+3=11")
    result = solve(parsed)
    assert result["problem_type"] == "linear"
    assert result["methods"][0].result_text == "4"


def test_linear_equation_with_parentheses():
    parsed = parse("3(x-1)=2x+4")
    result = solve(parsed)
    assert result["problem_type"] == "linear"
    assert result["methods"][0].result_text == "7"


def test_quadratic_factoring_and_formula():
    parsed = parse("x^2-5x+6=0")
    result = solve(parsed)
    assert result["problem_type"] == "quadratic"
    method_names = [m.name for m in result["methods"]]
    assert any("факторизација" in n for n in method_names)
    assert any("формула" in n for n in method_names)
    for m in result["methods"]:
        assert "2" in m.result_text and "3" in m.result_text


def test_quadratic_no_real_roots():
    parsed = parse("x^2+x+1=0")
    result = solve(parsed)
    assert result["problem_type"] == "quadratic"
    # нема факторизација над рационални броеви за овој случај
    assert len(result["methods"]) == 1
    assert "комплексни" in result["methods"][0].result_text


def test_simplify_expression():
    parsed = parse("2x+3(x-1)")
    result = solve(parsed)
    assert result["problem_type"] == "simplify"
    assert result["methods"][0].result_text.replace(" ", "") in ("5*x-3", "5x-3")


def test_normalize_unicode_symbols():
    parsed = parse("2×x−4=0")
    result = solve(parsed)
    assert result["problem_type"] == "linear"
    assert result["methods"][0].result_text == "2"


def test_implicit_multiplication_sqrt():
    parsed = parse("sqrt9+1")
    result = solve(parsed)
    assert result["problem_type"] == "simplify"
    assert result["methods"][0].result_text == "4"


def test_latex_to_plain_basic_constructs():
    assert latex_to_plain(r"\frac{1}{2}+x") == "(1)/(2)+x"
    assert latex_to_plain(r"x^{2}-5x+6=0") == "x^(2)-5x+6=0"
    assert latex_to_plain(r"\sqrt{9}+1") == "sqrt(9)+1"
    assert latex_to_plain(r"\left(x+1\right)\times 2") == "(x+1)* 2"


def test_latex_to_plain_strips_spurious_matrix_wrapper():
    # Реален случај забележан од handwriting сервисот - моделот понекогаш
    # "гледа" вишок ред (пр. линија од хартијата) и го враќа изразот
    # завиткан во \begin{matrix}...\end{matrix} со празен втор ред.
    assert latex_to_plain(r"\begin{matrix}1+1\\ \end{matrix}") == "1+1"
    assert latex_to_plain(r"\begin{matrix}2x+3=11\\ \end{matrix}") == "2x+3=11"


def test_parse_latex_from_handwriting_model():
    parsed = parse_latex(r"1+1")
    result = solve(parsed)
    assert result["problem_type"] == "simplify"
    assert result["methods"][0].result_text == "2"

    parsed_eq = parse_latex(r"x^{2}-5x+6=0")
    result_eq = solve(parsed_eq)
    assert result_eq["problem_type"] == "quadratic"
