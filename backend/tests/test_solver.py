"""
Основни тестови за parser + solver pipeline-от.
Стартување (од backend/ папката): python -m pytest tests/ -v
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.expression_parser import ParseError, latex_to_plain, parse, parse_latex
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
    method_names = [m.name.lower() for m in result["methods"]]
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


def test_latex_matrix_hallucination_of_minus_reconstructed():
    # Повторлив реален образец (3+ случаи) - моделот го "гледа" минус
    # знакот како колонски разделувач (&) и враќа "4-3" / "10-4" како
    # матрица наместо равенка со минус. Апликацијава НИКОГАШ не поддржува
    # матрици, па секој ваков излез е сигурна халуцинација - безбедно е
    # автоматски да се реконструира минусот наместо рачна поправка.
    assert latex_to_plain(r"\begin{matrix}4&3\\ 4&3\end{matrix}") == "4-3"
    assert latex_to_plain(r"\begin{matrix}10&-4\\ 4\end{matrix}") == "10-4"

    parsed = parse_latex(r"\begin{matrix}10&-4\\ 4\end{matrix}")
    result = solve(parsed)
    assert result["methods"][0].result_text == "6"


def test_invalid_characters_raise_parse_error_not_silent_wrong_answer():
    # Реален случај - handwriting моделот целосно погрешно "прочитал"
    # "4-3" како 2x2 матрица '4&3 / 4&3'. Без валидацијата, "4&3" тивко
    # би се парсирало како Python bitwise-AND (4&3=0) и би дало лажно
    # веродостоен, но целосно погрешен резултат "0" наместо грешка.
    for bad_input in ["4&3", "3~3", "4#3", "x|y"]:
        try:
            parse(bad_input)
            assert False, f"{bad_input!r} требаше да фрли ParseError"
        except ParseError:
            pass


def test_strips_trailing_ellipsis_artifact():
    # Реален случај - handwriting моделот врати "6+6..." (trailing точки,
    # генеративен артефакт) наместо чисто "6+6".
    parsed = parse("6+6...")
    assert parsed.normalized_text == "6+6"
    result = solve(parsed)
    assert result["methods"][0].result_text == "12"


def test_parse_latex_from_handwriting_model():
    parsed = parse_latex(r"1+1")
    result = solve(parsed)
    assert result["problem_type"] == "simplify"
    assert result["methods"][0].result_text == "2"

    parsed_eq = parse_latex(r"x^{2}-5x+6=0")
    result_eq = solve(parsed_eq)
    assert result_eq["problem_type"] == "quadratic"


def test_derivative_polynomial():
    parsed = parse("diff(x^2+3x,x)")
    assert parsed.kind == "derivative"
    result = solve(parsed)
    assert result["problem_type"] == "derivative"
    assert result["methods"][0].result_text == "2*x + 3"


def test_derivative_trig():
    parsed = parse("diff(sin(x),x)")
    result = solve(parsed)
    assert result["methods"][0].result_text == "cos(x)"


def test_integral_polynomial():
    parsed = parse("integrate(x^2,x)")
    assert parsed.kind == "integral"
    result = solve(parsed)
    assert result["problem_type"] == "integral"
    assert result["methods"][0].result_text == "C + x**3/3"


def test_integral_trig():
    parsed = parse("integrate(sin(x),x)")
    result = solve(parsed)
    assert result["methods"][0].result_text == "C - cos(x)"


def test_linear_system_2x2():
    parsed = parse("x+y=5;x-y=1")
    assert parsed.kind == "system"
    assert len(parsed.variables) == 2
    result = solve(parsed)
    assert result["problem_type"] == "system"
    result_text = result["methods"][0].result_text
    assert "x = 3" in result_text and "y = 2" in result_text


def test_basic_trig_equation():
    parsed = parse("sin(x)=0.5")
    result = solve(parsed)
    assert result["problem_type"] == "trigonometric"
    assert "2" in result["methods"][0].result_text  # 2*pi*n период


def test_cubic_factoring():
    parsed = parse("x^3-6x^2+11x-6=0")
    result = solve(parsed)
    assert result["problem_type"] == "higher_degree"
    result_text = result["methods"][0].result_text
    for root in ("1", "2", "3"):
        assert root in result_text


def test_cubic_general_fallback():
    # Не се факторизира убаво над рационални броеви - паѓа на општо
    # решение, сепак мора да врати нешто (не грешка/crash)
    parsed = parse("x^3+x+1=0")
    result = solve(parsed)
    assert result["problem_type"] == "higher_degree"
    assert result["methods"][0].result_text


def test_trig_without_parens_defaults_to_degrees():
    # Реален случај - "sin30" (без загради) погрешно се делеше на s*i*n*30
    # (implicit multiplication). Сега автоматски добива загради И се
    # третира како степени (школски стандард), не радијани.
    for text, expected in [("sin30", "1/2"), ("cos60", "1/2"), ("tan45", "1")]:
        parsed = parse(text)
        result = solve(parsed)
        assert result["methods"][0].result_text == expected, text


def test_trig_with_pi_stays_radians():
    parsed = parse("sin(pi/6)")
    result = solve(parsed)
    assert result["methods"][0].result_text == "1/2"


def test_trig_symbolic_argument_stays_radians_for_calculus():
    # sin(x) во извод МОРА да остане во радијани - inaku cos(x) правилото
    # за извод не важи
    parsed = parse("diff(sin(x),x)")
    result = solve(parsed)
    assert result["methods"][0].result_text == "cos(x)"
