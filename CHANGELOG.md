# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.2] - 2026-05-21

### Fixed

- `Lingo._build_filters` (the `@app.when` reflexive-pattern router)
  built a dynamic `Filter` model with auto-named fields
  (`option_0`, `option_1`, …) but then looked those names up as keys
  in `self._filters`, which is keyed by condition strings — every
  triggered filter raised `KeyError`. Anyone using `@app.when` in
  lingo 1.x or 2.0.1 was silently broken. Fixed by capturing the
  ordered list of condition keys and indexing by position.

### Added

- New example `examples/wizard.py` — the README quickstart turned
  into a runnable file. Demonstrates the four conversational-modeling
  primitives (`engine.ask`, `engine.choose`, `engine.decide`,
  `engine.create`).
- `tests/test_examples.py` — every example in `examples/` is now
  driven end-to-end against `MockLLM` in CI. No more silent bitrot.
- Examples catalog section in the README + cross-link from
  `docs/user-guide.md` Recipe 1 to `wizard.py`.

### Changed

- All examples refactored to expose a sync `main()` entrypoint
  (with `if __name__ == "__main__": main()`) so they're import-safe
  for tests. `examples/__init__.py` sets a dummy `API_KEY` env var
  to satisfy the `openai.AsyncOpenAI` constructor at module load
  without real network calls.
- `examples/banker.py` no longer passes a `ToolResult` (BaseModel)
  through `Message.system(...)` (which only accepts `str` and
  crashed) — the BaseModel is now passed directly to
  `engine.reply()`, which handles it correctly via
  `_expand_content`.

## [2.0.1] - 2026-05-21

### Fixed

- `Message.model_dump()` now serializes `tool_calls` on assistant
  messages and `tool_call_id` on tool-role messages (both REQUIRED by
  OpenAI's chat completions API when replaying a tool-using
  conversation). Caught by an end-to-end live test in the
  lovelaice integration; would have broken any consumer doing native
  tool-calling round-trips against real models. Regression tests in
  `tests/test_message_model_dump.py`.
- `Engine.put(msg)` was writing to the wrong queue (`_signal_queue`
  instead of `_input_queue`), causing `Lingo.chat()` to hang
  indefinitely on the resume path when a skill paused at
  `engine.ask()` / `engine.input()`. Regression test in
  `tests/test_engine.py`.
- `Lingo.chat()` context-sync was off by one — the prepended system
  message shifted the slice, causing the last user message to be
  duplicated on every turn. Regression tests in
  `tests/test_core_lingo.py`.

### Other

- Coverage push: `core.py` 21%→96%, `cli.py` 0%→100%,
  `engine.py` 45%→100%, total 55%→74%.
- New examples: `examples/native_tool_call.py` (manual dispatch loop)
  and `examples/native_tool_call_streaming.py` (streaming callbacks).
- README + `docs/user-guide.md` Recipe 8 cover the new native path.

## [2.0.0] - 2026-05-21

### Breaking

- `Message` shape changed. Adds optional fields `tool_calls`,
  `thinking`, `stop_reason`. Consumers that pattern-match on `Message`
  fields may need adjustment.

### Added

- `ToolCall` model (`id`, `name`, `arguments: dict`).
- `LLM.chat(tools=[...])` accepts a list of `lingo.Tool` objects and
  serializes their schemas into the OpenAI native `tools=[...]` API
  field via `tool_to_openai_schema()`. Returned `Message.tool_calls` is
  populated when the model emits tool calls.
- Streaming callbacks `on_toolcall_start(call_id, name)`,
  `on_toolcall_delta(call_id, cumulative_args_so_far)`,
  `on_toolcall_end(call_id, args)` mirror the existing `on_token` /
  `on_reasoning_token` pattern.
- `Message.thinking` accumulates the streamed reasoning fragments
  (previously only available via the `on_reasoning_token` callback).
- `Message.stop_reason` captures the OpenAI `finish_reason`
  (`stop` / `length` / `tool_calls` / `content_filter`).
- `examples/native_tool_call.py` — end-to-end manual dispatch loop
  demonstrating the new native tool-calling path.
- `examples/native_tool_call_streaming.py` — same flow with streaming
  callbacks (`on_toolcall_start/delta/end`) for live UI rendering.

## [1.5.0] - 2026-05-07

### Added
- **Reasoning passthrough on `LLM.chat()`.** New constructor args
  `on_reasoning_token` (callback) and `reasoning` (OpenRouter body
  kwarg, e.g. `{"effort": "high"}`). Streamed `delta.reasoning` /
  `delta.reasoning_content` / `delta.thoughts` fragments — including
  fields the OpenAI SDK keeps under `model_extra` — are forwarded to
  the callback. The `reasoning` body kwarg is injected only on the
  streaming `chat()` path; `create()` (structured output via OpenAI
  `parse()`) never carries it, since `parse()` rejects unknown kwargs.

## [1.4.1] - 2026-03-13

### Changed
- Replaced `black` with `ruff` for linting and formatting across the codebase.
- Updated `makefile` and CI/CD pipelines to use `ruff`.

### Fixed
- Addressed various linting issues discovered by `ruff`, including unused imports and bare `except` blocks.

## [1.4.0] - 2026-03-13

### Added
- **Comprehensive Documentation Suite:** A new `/docs` directory containing in-depth guides on Architectural Design, Deployment, Development, and a full User Guide.
- **AI Agent Skill Guide:** Specialized documentation (`docs/skill.md`) designed to instruct AI coding agents on how to use Lingo-AI idiomatically.
- **CI/CD Documentation Workflow:** Automated MkDocs rendering and GitHub Pages deployment on every release.
- **MkDocs Integration:** Added `mkdocs-material` as a development dependency and configured `mkdocs.yml`.

### Changed
- Restored original library-focused `README.md` to preserve project identity post-Gemini integration.

## [0.11.0] - 2026-03-11

### Added
- Unified `install.sh` script that handles both initial project bootstrapping and non-destructive framework updates/integrations in existing repositories.
- Automatic git environment validation (clean tree requirement) and post-install commits to `install.sh`.
- Interactive summary and confirmation of proposed changes (created vs. updated files) in the installer.
- **Documentation Suite:** Integrated MkDocs with Material theme and a comprehensive User Guide based on "The Architect in the Machine" philosophy.
- **Operational Safety:** Conditional `make` and journal hook execution based on file modification times.
- **Performance:** Simplified `/onboard` command using direct file analysis instead of sub-agents.
- **Refinement:** Strictly enforced non-execution mandate for `/plan` output.

### Removed
- `add-gemini.sh` script (its logic is now integrated into the unified `install.sh`).

### Changed
- Streamlined `README.md` with a single, unified "Quick Start" command for all use cases.
- Relocated `install.sh` to `docs/` to enable serving via GitHub Pages at `https://apiad.github.io/starter/install.sh`.

## [0.10.1] - 2026-03-03

### Changed
- Updated `install.sh` and `add-gemini.sh` to suggest running `gemini /onboard` instead of launching the CLI automatically.

### Fixed
- Improved ASCII art and version display in the banner function of installer scripts.

## [0.10.0] - 2026-03-03

### Added
- Implemented `install.sh` scaffolding script to automate project bootstrapping from the template.
- Implemented `add-gemini.sh` integration script for adding the framework to existing repositories.
- Added professional ASCII banners with versioning to all installer scripts for improved UX.

### Changed
- Refined `README.md` with streamlined Quick Start and new Integration sections.

## [0.9.0] - 2026-03-03

### Added
- Implemented `/debug` command and forensic `debugger` subagent for root-cause analysis (RCA).
- Created a specialized forensic investigation workflow for bug detection and documentation.

### Changed
- Refined `scaffold` and `revise` command instructions for better clarity and consistency.
- Improved README documentation and project metadata.

## [0.8.0] - 2026-03-02

### Added
- Implemented `/plan` command workflow and `planner` subagent for architectural planning.
- Added `/draft` and `/revise` commands with `reporter` and `editor` subagents for structured content generation.
- Created an actionable project style guide in `.gemini/style-guide.md`.
- Drafted the "The Architect in the Machine" Substack article showcasing the framework.

### Changed
- Overhauled the `/research` command into an extensible, executive-style reporting workflow with iterative updates and asset linking.
- Refactored `/draft` and `/revise` commands to integrate step-by-step, style-driven audits.
- Refined `/plan` and `/task` workflows.

## [0.7.0] - 2026-03-02

### Added
- Implemented `/cron` command with `cron.toml` task configuration using systemd user timers for background execution.
- Project badges and emoji headers to `README.md` for better visual appeal.

### Changed
- Comprehensive overhaul of `README.md` with detailed command descriptions and common workflows.
- Cleaned up `.gemini/settings.json` by removing redundant plan path.

## [0.6.0] - 2026-03-02

### Added
- Automate `gh release create` in `/release` command.

### Changed
- Cleanup deleted maintenance typo file.
- Fix maintenance typo and simplify command descriptions.
- Update `/research` command and subagents for better organization.

### Fixed
- Remove unused `subproc` import in `welcome` hook.

## [0.5.0] - 2026-02-28

### Added
- New `/issues` command to manage project issues with GitHub CLI, supporting summaries, creation/updates, and work modes.

### Changed
- Updated `README.md` and `welcome.py` hook to include information about the `/issues` command.
- Internal documentation updates in `TASKS.md` and `journal/`.

## [0.4.0] - 2026-02-28

### Changed
- Refactored the hook system: centralized shared logic into `.gemini/hooks/utils.py` and renamed hook files for consistency (`session.py`, `log.py`, `make.py`, `journal.py`).
- Added PEP 257 docstrings to all hooks and improved internal documentation.

### Added
- Updated `TASKS.md` and `journal/2026-02-28.md` with details of the hook refactoring.
- Ignore `__pycache__` directories in `.gitignore`.

## [0.3.0] - 2026-02-28

### Added
- Refactored the `/research` command into a robust 3-phase workflow using specialized `researcher` and `reporter` subagents.
- Rewrote the `README.md` to explain the opinionated framework, the agent's behavior, hooks, commands, journaling, and the project initialization workflow.

### Changed
- Improved the `/release` command logic to include `README.md` updates and fix formatting issues.

## [0.2.0] - 2026-02-28

### Added
- Added a new `/scaffold` command for project initialization with modern tooling (e.g., Python/uv, JS/npm).

### Changed
- Updated the `/release` command to include updating `README.md` and fixed a minor typo.

## [0.1.0] - 2026-02-28

### Added
- Consolidated `/task/*` commands into a single `/task` command in `.gemini/commands/task.toml`.
- Enhanced `/release` command to include version update steps.
- Initial project task tracking with `TASKS.md`.
- Daily journal tracking in `journal/`.

### Changed
- Refactoried `.gemini/hooks/welcome.py` to include new commands.
- Simplified `GEMINI.md` to a cleaner starter template.
- Updated `.gemini/commands/release.toml` to include dependency and source version updates.

### Fixed
- Typo in `release.toml` formatting.
