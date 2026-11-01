"""Name-to-factory registries shared by built-in and teammate-defined transforms."""

from collections.abc import Callable


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self.factories: dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(factory):
            if name in self.factories:
                raise ValueError(f"Duplicate {self.kind}: {name}")
            self.factories[name] = factory
            return factory

        return decorator

    def build(self, steps: list[dict]) -> list[Callable]:
        result = []
        for step in steps:
            options = dict(step)
            name = options.pop("name")
            if not options.pop("enabled", True):
                continue
            if name not in self.factories:
                raise ValueError(
                    f"Unknown {self.kind} '{name}'. Available: {sorted(self.factories)}"
                )
            result.append(self.factories[name](**options))
        return result
