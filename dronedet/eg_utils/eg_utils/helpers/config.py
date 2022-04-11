import importlib


def import_object(name: str) -> object:
    components = name.split(".")
    mod = importlib.import_module(".".join(components[:-1]))
    return getattr(mod, components[-1])
