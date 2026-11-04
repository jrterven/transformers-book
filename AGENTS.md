# Repository workflow

- This is the public code repository for *Transformers*. Keep manuscript
  chapters and bibliography databases in the private
  `jrterven/transformers-latex` repository.
- Before changing code, pull the latest default branch and reconcile local
  changes. When working through the GitHub connector rather than a Git
  checkout, fetch the current branch and tree, compare every tracked blob,
  and retrieve newer files before editing.
- Generate the book's excerpts with `python scripts/export_snippets.py`;
  check them with `python scripts/export_snippets.py --check`. Copy changed
  excerpts to `transformers-book/snippets/` in the private LaTeX repository.
- Run the relevant examples and tests before updating GitHub. Check the remote
  head again before publishing; never overwrite another author's commits.
- Access the book's Overleaf project through the connected `overleaf` MCP
  server, project `transformers`. Prefer its supported tools over computer use.
  Keep manuscript edits in the private repository, reconcile Overleaf changes
  before editing, and synchronize GitHub and Overleaf deliberately.
