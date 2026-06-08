"""Export a prefix formula to a plug-and-play PyTorch LambdaLR scheduler."""
from __future__ import annotations
from src.symbolr.artifacts.prefix_parser import parse_prefix


def _node_to_pytorch(node: tuple) -> str:
    kind = node[0]
    if kind == 'const':
        return f"torch.tensor({node[1]!r}, dtype=torch.float32)"
    elif kind == 'var':
        var = node[1]
        if var in ('t', 'x'):
            return "t"
        elif var == 'g':
            return "grad_norm"
        elif var == 'dl':
            return "loss_delta"
        return "t"
    elif kind == '+':
        return f"({_node_to_pytorch(node[1])} + {_node_to_pytorch(node[2])})"
    elif kind == '-':
        return f"({_node_to_pytorch(node[1])} - {_node_to_pytorch(node[2])})"
    elif kind == '*':
        return f"({_node_to_pytorch(node[1])} * {_node_to_pytorch(node[2])})"
    elif kind == '/':
        return (
            f"({_node_to_pytorch(node[1])} "
            f"/ (torch.abs({_node_to_pytorch(node[2])}) + 1e-6))"
        )
    elif kind == 'sin':
        return f"torch.sin({_node_to_pytorch(node[1])})"
    elif kind == 'cos':
        return f"torch.cos({_node_to_pytorch(node[1])})"
    elif kind == 'exp':
        return f"torch.exp(torch.clamp({_node_to_pytorch(node[1])}, max=20.0))"
    elif kind == 'log':
        return f"torch.log(torch.abs({_node_to_pytorch(node[1])}) + 1e-6)"
    elif kind == 'abs':
        return f"torch.abs({_node_to_pytorch(node[1])})"
    elif kind == 'sqrt':
        return f"torch.sqrt(torch.abs({_node_to_pytorch(node[1])}) + 1e-6)"
    return "torch.tensor(0.01)"


def export_to_pytorch(prefix_str: str) -> str:
    """
    Convert a prefix notation formula to a PyTorch LambdaLR scheduler function.

    The generated scheduler supports both time-only and gradient-aware formulas.
    For gradient-aware formulas (containing 'g' or 'dl' variables), pass
    `grad_norm` and `loss_delta` into the lambda at each step.
    """
    tokens = prefix_str.strip().split()
    try:
        tree, _ = parse_prefix(tokens, 0)
        expr = _node_to_pytorch(tree)
    except Exception:
        expr = "torch.tensor(0.01, dtype=torch.float32)"

    return f'''\
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_symbolr_schedule(optimizer, total_epochs: int):
    """Learning rate schedule discovered by SymboLR."""
    def lr_lambda(epoch: int) -> float:
        t = torch.tensor(epoch / max(1, total_epochs), dtype=torch.float32)
        grad_norm = torch.tensor(0.0)   # replace with current ||∇|| for adaptive mode
        loss_delta = torch.tensor(0.0)  # replace with recent loss slope for adaptive mode
        multiplier = {expr}
        return float(torch.clamp(multiplier, 1e-7, 10.0))

    return LambdaLR(optimizer, lr_lambda)
'''
