"""Strategy registration and lookup."""

from importlib import import_module

_REGISTRY = {}


def register(name):
    def decorator(strategy):
        _REGISTRY[name] = strategy
        return strategy
    return decorator


def get_strategy(name):
    if name not in _REGISTRY:
        try:
            import_module(f"strategies.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"strategies.{name}":
                raise
    if name not in _REGISTRY:
        names = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"unknown strategy {name!r}; registered strategies: {names}")
    strategy = _REGISTRY[name]
    return strategy() if isinstance(strategy, type) else strategy


def registered():
    return dict(_REGISTRY)
