# Contributing

Contributions are welcome — bug fixes, new summarizer types, improved prompts, or documentation.

## Development Setup

```bash
git clone https://github.com/jmars/vibe-summarizer.git
cd vibe-summarizer

python -m venv venv
source venv/bin/activate    # or venv\Scripts\activate on Windows

pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=vibe_summarizer
```

## Linting

```bash
pip install ruff
ruff check src/ tests/
```

## Project Structure

```
src/vibe_summarizer/
├── __init__.py      # Package metadata
├── __main__.py      # CLI entry points
├── llm.py           # Provider-agnostic LLM client
├── session.py       # Session summarizer
└── transcript.py    # Transcript summarizer
```

## License

This project is licensed under the [MIT License](LICENSE). By contributing, you agree that your contributions will be licensed under the same license.
