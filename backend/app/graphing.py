
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy
from sympy import Symbol

from .expression_parser import ParsedProblem

_X_RANGE = (-10, 10)
_SAMPLES = 400


def _lambdify_safe(expr, var):
    return sympy.lambdify(var, expr, modules=["numpy"])


def _new_figure():
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=130)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.axvline(0, color="#888", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True)
    plt.close(fig)
    return buf.getvalue()


def render_single_expression(expr, var: Symbol) -> bytes:
    """График на y = expr(x)."""
    f = _lambdify_safe(expr, var)
    xs = np.linspace(*_X_RANGE, _SAMPLES)
    with np.errstate(all="ignore"):
        ys = f(xs)
    ys = np.asarray(ys, dtype=float)
    ys = np.where(np.abs(ys) > 1e4, np.nan, ys)  # исфрли асимптоти/поли

    fig, ax = _new_figure()
    ax.plot(xs, ys, color="#2563eb", linewidth=2)
    ax.set_title(f"y = {sympy.pretty(expr, use_unicode=False)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return _fig_to_png(fig)


def render_equation_with_roots(lhs, rhs, var: Symbol, roots: list) -> bytes:

    f_lhs = _lambdify_safe(lhs, var)
    f_rhs = _lambdify_safe(rhs, var)
    xs = np.linspace(*_X_RANGE, _SAMPLES)

    def _eval(f, xs):

        y = f(xs)
        return np.broadcast_to(np.asarray(y, dtype=float), xs.shape).copy()

    with np.errstate(all="ignore"):
        ys_lhs = _eval(f_lhs, xs)
        ys_rhs = _eval(f_rhs, xs)

    fig, ax = _new_figure()
    ax.plot(xs, ys_lhs, color="#2563eb", linewidth=2, label="лева страна")
    ax.plot(xs, ys_rhs, color="#f97316", linewidth=2, label="десна страна")

    for r in roots:
        try:
            rf = float(r)
        except (TypeError, ValueError):
            continue
        yf = float(f_lhs(rf))
        ax.plot(rf, yf, "o", color="#16a34a", markersize=8, zorder=5)
        ax.annotate(f"x={rf:.3g}", (rf, yf), textcoords="offset points", xytext=(6, 6))

    ax.legend(loc="best", fontsize=8)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return _fig_to_png(fig)


def build_graph(parsed: ParsedProblem, solve_result: dict) -> bytes | None:

    try:
        if parsed.kind in ("expression", "derivative", "integral") and len(parsed.variables) == 1:
            return render_single_expression(parsed.sympy_obj, parsed.variables[0])

        if parsed.kind == "equation" and len(parsed.variables) == 1:
            var = parsed.variables[0]

            try:
                roots = sympy.solve(parsed.sympy_obj, var)
            except Exception:
                roots = []
            return render_equation_with_roots(
                parsed.sympy_obj.lhs, parsed.sympy_obj.rhs, var, roots
            )
    except Exception:
        return None

    return None
