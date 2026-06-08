"""
Canonical prefix notation formula parser for SymboLR.

This is the single source of truth for prefix formula parsing.
All exporters (PyTorch, LaTeX) and evaluators derive from this module.
Supports variables: t (normalized time), g (gradient norm), dl (loss slope).
"""
from __future__ import annotations
import math
from typing import Any

BINARY_OPS = frozenset({'+', '-', '*', '/'})
UNARY_OPS  = frozenset({'sin', 'cos', 'exp', 'log', 'abs', 'sqrt'})
VARIABLES  = frozenset({'t', 'x', 'g', 'dl'})  # x is a legacy alias for t

# Safe value bounds for LR output
LR_MIN = 1e-7
LR_MAX = 10.0


def parse_prefix(tokens: list[str], pos: int = 0) -> tuple[Any, int]:
    """
    Recursively parse prefix notation tokens into an AST node.

    Returns:
        (node, next_pos) where node is a tuple describing the expression tree.
        Leaf nodes: ('const', float) or ('var', str)
        Unary nodes: (op_str, child_node)
        Binary nodes: (op_str, left_node, right_node)
    """
    if pos >= len(tokens):
        return ('const', 0.0), pos + 1

    tok = tokens[pos]

    if tok in BINARY_OPS:
        left, pos = parse_prefix(tokens, pos + 1)
        right, pos = parse_prefix(tokens, pos)
        return (tok, left, right), pos
    elif tok in UNARY_OPS:
        arg, pos = parse_prefix(tokens, pos + 1)
        return (tok, arg), pos
    elif tok in VARIABLES:
        return ('var', tok), pos + 1
    else:
        try:
            return ('const', float(tok)), pos + 1
        except ValueError:
            return ('const', 0.01), pos + 1


def evaluate_tree(node: tuple, bindings: dict[str, float]) -> float:
    """
    Evaluate a parsed AST node with the given variable bindings.

    All operations are numerically safe: division by zero, log of negative,
    and exp overflow are handled without raising exceptions.
    """
    kind = node[0]

    if kind == 'const':
        return float(node[1])
    elif kind == 'var':
        var = 't' if node[1] == 'x' else node[1]
        return float(bindings.get(var, 0.0))
    elif kind == '+':
        return evaluate_tree(node[1], bindings) + evaluate_tree(node[2], bindings)
    elif kind == '-':
        return evaluate_tree(node[1], bindings) - evaluate_tree(node[2], bindings)
    elif kind == '*':
        return evaluate_tree(node[1], bindings) * evaluate_tree(node[2], bindings)
    elif kind == '/':
        denom = evaluate_tree(node[2], bindings)
        return evaluate_tree(node[1], bindings) / (abs(denom) + 1e-6)
    elif kind == 'sin':
        return math.sin(evaluate_tree(node[1], bindings))
    elif kind == 'cos':
        return math.cos(evaluate_tree(node[1], bindings))
    elif kind == 'exp':
        return math.exp(min(evaluate_tree(node[1], bindings), 20.0))
    elif kind == 'log':
        val = evaluate_tree(node[1], bindings)
        return math.log(abs(val) + 1e-6)
    elif kind == 'abs':
        return abs(evaluate_tree(node[1], bindings))
    elif kind == 'sqrt':
        val = evaluate_tree(node[1], bindings)
        return math.sqrt(abs(val) + 1e-6)
    return 0.0


def evaluate_formula(prefix_str: str, **bindings: float) -> float:
    """
    Evaluate a prefix formula string with given variable bindings.

    The result is clamped to [LR_MIN, LR_MAX] to ensure valid LR output.

    Example:
        evaluate_formula("cos * 3.14159 t", t=0.5)
        evaluate_formula("* exp neg_g cos t", t=0.5, g=-0.2)
    """
    tokens = prefix_str.strip().split()
    if not tokens:
        return LR_MIN
    try:
        tree, _ = parse_prefix(tokens, 0)
        val = evaluate_tree(tree, bindings)
        if not math.isfinite(val):
            return LR_MIN
        return float(max(LR_MIN, min(LR_MAX, val)))
    except Exception:
        return LR_MIN
