"""Export a prefix formula to LaTeX notation."""
from __future__ import annotations
from src.symbolr.artifacts.prefix_parser import parse_prefix

def _node_to_latex(node: tuple) -> str:
    kind = node[0]
    if kind == 'const':
        val = float(node[1])
        return str(round(val, 4))
    elif kind == 'var':
        var = node[1]
        if var in ('t', 'x'):
            return "t"
        elif var == 'g':
            return r"\|g\|"
        elif var == 'dl':
            return r"\Delta\ell"
        return "t"
    elif kind == '+':
        return rf"\left( {_node_to_latex(node[1])} + {_node_to_latex(node[2])} \right)"
    elif kind == '-':
        return rf"\left( {_node_to_latex(node[1])} - {_node_to_latex(node[2])} \right)"
    elif kind == '*':
        return rf"\left( {_node_to_latex(node[1])} \cdot {_node_to_latex(node[2])} \right)"
    elif kind == '/':
        return rf"\frac{{{_node_to_latex(node[1])}}}{{{_node_to_latex(node[2])}}}"
    elif kind == 'sin':
        return rf"\sin\!\left( {_node_to_latex(node[1])} \right)"
    elif kind == 'cos':
        return rf"\cos\!\left( {_node_to_latex(node[1])} \right)"
    elif kind == 'exp':
        return rf"e^{{{_node_to_latex(node[1])}}}"
    elif kind == 'log':
        return rf"\ln\!\left( {_node_to_latex(node[1])} \right)"
    elif kind == 'abs':
        return rf"\left| {_node_to_latex(node[1])} \right|"
    elif kind == 'sqrt':
        return rf"\sqrt{{{_node_to_latex(node[1])}}}"
    return "0.01"


def export_to_latex(prefix_str: str) -> str:
    """Convert a prefix notation formula to a LaTeX math string."""
    tokens = prefix_str.strip().split()
    try:
        tree, _ = parse_prefix(tokens, 0)
        expr = _node_to_latex(tree)
    except Exception:
        expr = "0.01"
    return f"$$ {expr} $$"
