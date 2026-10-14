# Coding Standards — ResoGenre

## Python Style
- Follow PEP 8 strictly.
- Maximum line length: 100 characters.
- Use `pathlib.Path` for all file path operations, never raw string concatenation.
- Prefer f-strings over `.format()` or `%` formatting.

## Imports
- Standard library imports first, then third-party, then local modules.
- Use absolute imports (e.g., `from src.model import GenreCNN`).
- Never use wildcard imports (`from module import *`).

## Error Handling
- Wrap file I/O and audio loading in try/except blocks.
- Log errors with meaningful messages including the file path that failed.
- Use `logging` module, not `print()`, for status messages in production code.

## Testing & Reproducibility
- Pin all random seeds to `42` at the start of every script.
- Use `torch.manual_seed`, `np.random.seed`, `random.seed`, and `torch.backends.cudnn.deterministic = True`.
- Document all hyperparameters in `config.py`, never hardcode in training loops.

## Git Practices
- Never commit model checkpoints (`.pth`), raw data, or processed tensors.
- Use `.gitignore` to exclude `data/`, `models/`, `outputs/`, `__pycache__/`.
- Commit messages: `[phase] description` (e.g., `[phase-1] add mel-spectrogram extraction`).

## Documentation
- Every module must start with a module-level docstring.
- Document tensor shapes in comments (e.g., `# [B, 1, 128, 130]`).
- Update `IMPLEMENTATION.md` status after completing any task.
