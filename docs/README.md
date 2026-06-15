# docs/ — external user-facing documentation

This folder holds material *for the people consuming this repo*:

- **API reference** (for library repos) — how to call the public functions / classes, what they accept, what they return, error modes.
- **Integration guides** (for library repos) — "I want to use this from `egm-classifier`'s training loop, how do I plug it in?" worked example.
- **Quick-start** (for executable / CLI repos) — install, then run-the-thing-end-to-end in five minutes.
- **Command reference** (for CLI repos) — every subcommand, its flags, what it does.
- **GUI walkthroughs** (for desktop apps like `egm-studio`) — screenshots + workflows.

**What does NOT go here:**

- Design rationale, "why we did it this way" — that's `project/`.
- Investigation reports — those live in the meta repo's `project/`.
- Repo-root `README.md` content — keep `docs/` for material too long or too detailed to live in the README. The README is the pitch + the install + the link to here.

**Conventions:**

- Markdown by default. If a static-site generator gets added later (MkDocs, Sphinx), this folder is the source root.
- Audience-first ordering: assume the reader doesn't know what's where. Start each page with "what is this and who is it for."
- Code examples are runnable. Tested-in-CI is ideal but defer until the docs are big enough to warrant the infrastructure.
- Don't pre-fill this folder before you have real users (= someone other than you, or you-in-three-months who has forgotten the codebase). Sparse is fine; misleading is not.

See the [`feedback-docs-vs-project-folders`](https://github.com/myocard-labs/intracardiac-platform/blob/main/project/) memory convention for the canonical rules.
