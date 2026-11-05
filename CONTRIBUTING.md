# Contributing Guide (CONTRIBUTING.md)

Thank you for your interest in contributing to the CartPole Reinforcement Learning repository! To keep the codebase robust, well-documented, and fully reproducible, please read and follow these contribution guidelines before submitting code.

---

## 1. Development Workflow

1. **Fork and Clone the Repository**:
   ```bash
   git clone https://github.com/orias/BOT-Reinforce-Learn-Cartpole.git
   cd BOT-Reinforce-Learn-Cartpole
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux or macOS:
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

3. **Create a Feature Branch**:
   - Use descriptive branch prefixes:
     - `feature/description` for new features or algorithms
     - `fix/description` for bug fixes
     - `docs/description` for documentation improvements
     - `refactor/description` for code restructuring

---

## 2. Coding and Commenting Standards

All contributions must strictly follow the rules documented in [COMMENT.md](COMMENT.md):

1. **Zero Inline Comments**:
   - Never write comments inside function bodies or at the end of lines with code.
   - Code must be self-explanatory through clean design and clear variable names.
2. **Module-Level Docstrings**:
   - Every Python file must start with a descriptive top-level docstring outlining purpose, CLI commands, and input/output artifacts.
3. **Structured Function Docstrings**:
   - Every function and class must include a structured docstring with `Input:`, `Output:`, and `Flow:` sections.
4. **Default Constants at the Top**:
   - Hyperparameters, thresholds, and paths must be defined as `DEFAULT_*` constants at the top of the file and passed as default arguments to functions.
5. **No Em-Dashes**:
   - Do not use the Unicode em-dash character (U+2014) in any documentation, comments, commit messages, or source code. Use colons, commas, periods, parentheses, or conjunctions instead.

---

## 3. Testing and Verification Requirements

Before opening a pull request, you must run and pass both of the following test suites locally:

1. **Unit Tests**:
   ```bash
   python -m pytest tests/test_dqn_units.py -v
   ```
   Requirement: All 4/4 tests must pass (**PASSED**).

2. **Phase Acceptance Gates**:
   ```bash
   python scripts/check_gate.py all
   ```
   Requirement: All 71/71 acceptance items must pass (**PASS**).

If any checks fail, resolve the issues before committing or submitting a pull request.

---

## 4. Commit Message Conventions

We follow Conventional Commits to maintain a clean git history:
- `feat: add new feature or algorithm`
- `fix: resolve a bug or incorrect logic`
- `docs: update documentation or reports`
- `refactor: clean up code without changing functionality`
- `test: add or update unit tests`
- `chore: tooling or build script maintenance`

Example:
```bash
git commit -m "feat: add support for custom pole mass in environment wrapper"
```

---

## 5. Submitting a Pull Request

- Rebase your branch against the latest `main` branch before opening a PR.
- Fill out the pull request template completely (.github/pull_request_template.md).
- Ensure CI workflows pass on your branch.

---

## 6. Questions and Support

For questions, suggestions, or issues, please open an Issue on GitHub or reach out to: `orias@orias.app`.
