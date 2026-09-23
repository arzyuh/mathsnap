"""Регресиони тестови за чистење на LaTeX излезот од handwriting моделот.
Секој случај е реален, забележан при тестирање со камера/слика."""
import pytest

from app.expression_parser import ParseError, normalize_text, parse_latex


@pytest.mark.parametrize(
    "latex, expected",
    [
        (r"\frac{5+5}{5}", "(5+5)"),  # лажна дропка (дуплиран операнд)
        (r"\begin{matrix}10&-4\end{matrix}", "10-4"),  # лажна матрица
        (r"\overline{x+y=7;x-y=1}", "x+y=7;x-y=1"),  # wrapper команда
        (r"\mathbb{x^3-6x^2+11x-6=0}", "x^3-6x^2+11x-6=0"),
        (r"\cdot\cdot diff(x^{\wedge}2,x)", "diff(x^2,x)"),  # ^ прочитано како \wedge
        (r"x^{1}3-6x^{1}2+11x-6=0", "x^3-6x^2+11x-6=0"),  # ^ прочитано како ^{1}
        (r"3x\_{-7}=8", "3x=8"),  # escaped underscore
    ],
)
def test_hallucination_cleanup(latex, expected):
    assert parse_latex(latex).normalized_text == expected


def test_nested_frac_and_integral():
    parsed = parse_latex(r"\int\frac{dx}{\sqrt{1-(x^{2}-2x+1)}}")
    assert parsed.kind == "integral"
    assert parsed.normalized_text == "integrate(1/(sqrt(1-(x^(2)-2x+1))),x)"


def test_simple_indefinite_integral():
    parsed = parse_latex(r"\int x^{2}dx")
    assert parsed.kind == "integral"


def test_digit_one_misread_as_letter_i():
    assert normalize_text("i+1") == "1+1"


def test_letter_i_inside_function_names_is_untouched():
    assert normalize_text("sin30") == "sin(30)"
    assert normalize_text("integrate(x^2,x)") == "integrate(x^2,x)"


def test_legitimate_power_one_is_untouched():
    assert parse_latex(r"x^{1}+2=5").normalized_text == "x^(1)+2=5"


def test_invalid_characters_still_rejected():
    with pytest.raises(ParseError):
        normalize_text("4&3")
