# myocard-package-name

> One-sentence pitch: PACKAGE_DESCRIPTION.

Part of the [myocard-labs](https://github.com/myocard-labs) cardiac signal processing toolkit.

> [!NOTE]
> This file was scaffolded from
> [`myocard-labs/python-template`](https://github.com/myocard-labs/python-template).
> If you're looking at the template repo itself, see `TEMPLATE_USAGE.md` for the
> rename procedure. The template is for **library packages only** — meta repos
> (`intracardiac-platform`) and the paper repo (`intracardiac-paper`) have their
> own hand-built skeletons.

---

## Why

Two or three paragraphs answering:

- What problem does this package solve?
- Where does it sit in the broader pipeline (upstream/downstream packages)?
- Who would actually want to install it standalone?

Be concrete. Hiring managers and future collaborators read this section first.

---

## Install

From PyPI (when published):

```bash
pip install myocard-package-name
```

From source (during pre-1.0 iteration):

```bash
pip install git+https://github.com/myocard-labs/GITHUB_REPO_NAME.git
```

Editable install for development:

```bash
git clone https://github.com/myocard-labs/GITHUB_REPO_NAME.git
cd GITHUB_REPO_NAME
pip install -e ".[dev]"
pre-commit install
```

---

## Quick start

A copy-pasteable example that demonstrates the headline use case. Keep it small:

```bash
myocard-foo --input data/sample.h5 --output results/
```

If the package is library-only with no CLI, link to the Programmatic Usage section below.

---

## Programmatic usage

```python
import myocard_package_name

# A short example showing the primary public API.
# Prefer one tight working example over five partial ones.
```

Link to longer examples in `examples/` or to `intracardiac-platform/examples/` for the cross-package workflows.

---

## Tests

```bash
pytest                  # full suite
pytest --cov            # with coverage
ruff check .            # lint
ruff format --check .   # format check
mypy                    # type check
```

CI runs the same checks on Python 3.10, 3.11, and 3.12 — see `.github/workflows/ci.yml`.

---

## Project status

This package is part of the in-progress [myocard-labs](https://github.com/myocard-labs) refactor. Pre-1.0 — expect breaking changes across minor versions until the schema/API stabilizes. See `intracardiac-platform/project/project_plan.md` for roadmap.

---

## Citation

If you use this software in academic work, please cite:

```bibtex
@software{klein_myocard_package_name_2026,
  author  = {Klein, Daniel},
  title   = {myocard-package-name: PACKAGE_DESCRIPTION},
  year    = {2026},
  url     = {https://github.com/myocard-labs/GITHUB_REPO_NAME},
}
```

---

## License

MIT — see [LICENSE](LICENSE). Attribution requirements for data sources used by this package are listed in [NOTICE](NOTICE).
