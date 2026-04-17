# Agent Ground Rules

This file establishes the global settings, workflow rules, and architectural choices for all AI agents working on this repository. Please adhere to these guidelines strictly.

## Git & Workflow
- **Branching:** Always create and work within a separate feature branch when developing a new feature or making significant changes.
- **Committing:** Commit frequently locally within your feature branch. Only push to the remote repository when the work is finished and ready for review/merging.
- **Versioning:** We use Semantic Versioning (SemVer). Only bump version numbers in the `main` branch when a feature branch gets merged into `main`. Do not bump versions in feature branches.

## Python Ecosystem
- **Tooling:** We exclusively use the Astral `uv` Python ecosystem.
- **Project Structure:** Ensure the presence of `.python-version` and `pyproject.toml`. Dependencies and the virtual environment should be managed by `uv`.
- **Execution:** When running tests or scripts, use `uv run python` (e.g., `uv run pytest`).
- **Code Quality:** Liberally use available Language Server Protocols (LSPs) and tools like `ruff` and type checkers for linting, formatting, and type-checking.

## Architecture & Code Style
- **File Size & Context:** Keep files relatively small so they easily fit into context windows.
- **Modularity:** Separate concerns and maintain loose coupling. Do not hesitate to extract relatively standalone logic into its own module.
- **Formats:** Use Markdown for documentation and TOML for configurations.

## Frameworks & Infrastructure
- **Web APIs (Small):** Use `litestar` for small APIs (typically 2-3 endpoints, such as a basic CRUD interface).
- **Web APIs (Large):** Use `FastAPI` for larger, more complex APIs.
- **Model Context Protocol (MCP):** Use `FastMCP` for building MCP servers.
- **Middleware:** Use `Caddy` as a reverse proxy/middleware to handle Authentication, CORS, Rate Limiting, and similar concerns. Do not build these directly into the application code.

## Security
- **No Hardcoded Secrets:** NEVER put credentials, API keys, tokens, or passwords in the code. This prevents sensitive information from ending up in prompts, context, or version control. Always use environment variables or secure secret managers.
