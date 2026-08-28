# Contributing to RepoScore

Thank you for your interest in contributing to RepoScore! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Bugs
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Search existing issues first to avoid duplicates
- Provide clear reproduction steps

### Suggesting Features
- Use the [feature request template](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the use case and expected behavior

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Ensure code passes linting
6. Submit a PR with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/reposcore.git
cd reposcore

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt
pip install -e .
```

## Code Standards

- Follow PEP 8 style guide
- Use type hints where appropriate
- Write tests for new functionality
- Keep functions small and focused

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model_and_api.py -v

# Run with coverage
pytest --cov=. tests/
```

## ML Model Considerations

When modifying scoring logic:
- The ML model (`models/rf_model.pkl`) and vectorizers are pre-trained
- Changes to `scoring_engine.py` may require model retraining
- Run `test_scoring_divergence.py` to compare ML vs heuristic scores

## Questions?

Open a [discussion](https://github.com/your-org/reposcore/discussions) or ask in a PR.