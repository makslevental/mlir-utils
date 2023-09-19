

def jax_not_installed():
    try:
        from jaxlib import mlir

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def mlir_bindings_not_installed():
    try:
        import mlir

        # don't skip
        return False

    except ImportError:
        # skip
        return True


def aie_bindings_not_installed():
    try:
        import aie

        # don't skip
        return False

    except ImportError:
        # skip
        return True
