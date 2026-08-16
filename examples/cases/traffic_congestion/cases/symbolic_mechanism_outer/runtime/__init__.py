def main(argv=None):
    """Load the legacy runtime only when its CLI is actually invoked."""

    from .runner import main as run_main

    return run_main(argv)

__all__ = ["main"]
