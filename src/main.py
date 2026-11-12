"""Command-line entry point: python3 src/main.py or python3 -m src.main."""

if __package__ in (None, ""):
    # Direct execution puts src/ on sys.path; package imports need its parent.
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workflow import main


if __name__ == "__main__":
    main()
