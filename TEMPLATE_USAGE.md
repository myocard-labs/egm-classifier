# How to use this template

This is the canonical scaffolding for every Python **library package** under the
[`myocard-labs`](https://github.com/myocard-labs) GitHub org. When you spawn a new
repo from it (via GitHub's "Use this template" button, or by copying this folder),
do the four things below before your first commit.

## When NOT to use this template

This template assumes a Python library shape — `src/`, `pyproject.toml`, pytest,
ruff, mypy. The following repos under `myocard-labs` have different shapes and get
their own hand-built skeletons instead:

- **`intracardiac-platform`** — meta/docs/integration repo. No Python package
  inside. See `intracardiac-platform/Documentation/` and `intracardiac-platform/
  project/refactor_checklist.md` Phase 1 for the layout.
- **`intracardiac-paper`** — LaTeX paper repo. See Phase 7 of the refactor
  checklist for the layout.

Both reference the convention table below for naming but don't use any of the
Python scaffolding from this template.

## 1. Pick the three names

Per the locked-in naming convention (see
`intracardiac-platform/project/refactor_checklist.md` and the
`project-github-org-myocard` memory):

| Surface         | Convention                  | Example                  |
| --------------- | --------------------------- | ------------------------ |
| GitHub repo     | bare, dash-separated        | `egm-contracts`          |
| PyPI dist       | `myocard-` prefix, dashes   | `myocard-egm-contracts`  |
| Python import   | underscored                 | `myocard_egm_contracts`  |

## 2. Find/replace the placeholders

Run a project-wide find/replace in this order — placeholders are case-sensitive
and the order matters because some are substrings of others (do `_` form first).

| Placeholder              | Replace with example                              |
| ------------------------ | ------------------------------------------------- |
| `myocard_package_name`   | `myocard_egm_contracts` (underscore import name)  |
| `myocard-package-name`   | `myocard-egm-contracts` (PyPI dist name)          |
| `GITHUB_REPO_NAME`       | `egm-contracts` (bare repo name)                  |
| `PACKAGE_DESCRIPTION`    | One-line summary used in pyproject + README       |

Then rename the source directory:

```bash
git mv src/myocard_package_name src/myocard_egm_contracts
```

## 3. Trim the NOTICE file

`NOTICE` ships a kitchen-sink of attribution snippets. Delete every line that
doesn't apply to this package; keep only the ones for data sources the package
actually touches (IAFDB, Finitewave, etc.).

## 4. Documentation folders

The template ships two folders deliberately:

- **`project/`** — internal: design docs, decision records, repo-specific
  roadmap. Audience is the people building this repo.
- **`docs/`** — external: user-facing material — API reference for libraries,
  quick-start / CLI reference / GUI walkthroughs for executables.

Both folders ship with a `README.md` placeholder explaining their intent. Leave
the placeholder in place even if the folder is empty — the slot existing is
what makes the convention costless. Cross-component investigation reports go in
the **meta repo's** `project/` (not in any component repo); see the
`feedback-docs-vs-project-folders` memory for the canonical rules.

## 5. First-commit checklist

```bash
pip install -e ".[dev]"
pre-commit install
pre-commit run --all-files   # auto-fix anything pre-commit catches
pytest                       # smoke test should pass
ruff check .
ruff format --check .
mypy
git add .
git commit -m "scaffold from python-template"
git tag v0.1.0
git push --tags
```

## When to update this template

When you find yourself repeatedly fixing the same omission across two or more
spawned repos, fold the fix back into this template. Do not let templates drift
silently — every repo started from a stale template inherits the staleness.

## Deliberately NOT in this template

- `requirements.txt` / lock files — use the `[project.optional-dependencies] dev`
  table in `pyproject.toml` for dev dependencies. Lock files belong in
  `intracardiac-paper` (for reproducibility) and individual app repos, not
  libraries.
- `MkDocs` / Sphinx scaffolding — defer until a package has enough public API
  surface to warrant it.
- Container images / Dockerfiles — deferred per the project plan.
- Conda packaging — deferred.
- GitHub Actions release-to-PyPI workflow — add per-package when you're ready
  to publish to TestPyPI/PyPI, not as part of the base template.
