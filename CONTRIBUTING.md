# Contributing

Contributions that improve reproducibility, evaluation, documentation, robustness or platform support are welcome.

1. Fork the repository.
2. Create a focused branch.
3. Keep dataset files and model checkpoints out of Git.
4. Run `python scripts/verify_repo.py`.
5. Describe scientific changes and expected metric impact in the pull request.

Please do not submit changes that silently alter evaluation splits or use outer-test labels for model selection.
