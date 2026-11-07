# src/utils/optimizer_coverage.py

"""Check that an optimizer updates exactly the trainable parameters of a module.

A trainable parameter left out of every parameter group silently keeps its initial value; one listed twice gets
two updates per step. Frozen parameters must not be in the optimizer.
"""

from collections import Counter


def optimizer_coverage_problems(module, optimizer):
    """The names of trainable parameters missing from the optimizer, listed more than once, and frozen or foreign
    parameters in it. Empty lists mean full, exact coverage."""
    names = {}
    for name, p in module.named_parameters():
        names.setdefault(id(p), name)
    trainable = {id(p) for p in module.parameters() if p.requires_grad}
    listed = Counter(id(p) for group in optimizer.param_groups for p in group["params"])
    return {
        "missing": sorted(names[i] for i in trainable - set(listed)),
        "duplicated": sorted(names.get(i, "<foreign>") for i, c in listed.items() if c > 1),
        "not_trainable": sorted(names.get(i, "<foreign>") for i in set(listed) - trainable),
    }


def check_optimizer_coverage(module, optimizer):
    """Raise ValueError unless the optimizer holds every trainable parameter of `module` exactly once and nothing
    else."""
    problems = {k: v for k, v in optimizer_coverage_problems(module, optimizer).items() if v}
    if problems:
        raise ValueError(f"Optimizer does not cover the trainable parameters exactly: {problems}")
