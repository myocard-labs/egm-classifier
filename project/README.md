# project/ — internal design + investigations

This folder holds material *for the people building / maintaining this repo*:

- **Design docs** — the locked decisions for a phase or major component. Example: a `phase1_design.pdf` rendered from a build script (PDF + the LaTeX or markdown source live side-by-side).
- **Decision records** — short markdown files capturing a single decision and *why* it was made. ADR-style.
- **Roadmap notes** — internal planning that hasn't earned a place in `project_plan.md` at the meta repo yet, or is too repo-specific to belong there.

**What does NOT go here:**

- Cross-component investigations (e.g., `v1_iafdb_investigation.md`) — those live in the meta repo's `project/` because by definition they drove decisions in more than one component.
- API reference, user-facing tutorials, install/quickstart — those go in `docs/`.
- Disposable scratch notes from a single chat — keep those out of git unless they crystalize into a decision.

**Conventions:**

- File names are descriptive: `phase1_design.pdf`, `bipolar_extraction_decision.md`, `roadmap.md`.
- If a doc is rendered from a build script, the script lives next to it (`build_design_doc.py` next to `phase1_design.pdf`).
- Markdown is the default. PDFs only when you need typesetting (figures, equations).
- Cross-reference the meta repo's `project/project_plan.md` for the broader context; don't duplicate it here.

See the [`feedback-docs-vs-project-folders`](https://github.com/myocard-labs/intracardiac-platform/blob/main/project/) memory convention for the canonical rules.
