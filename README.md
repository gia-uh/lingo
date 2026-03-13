# 🤖 Gemini CLI Opinionated Framework

<div align="center">

[![Release](https://img.shields.io/github/v/release/apiad/starter?style=for-the-badge&color=blue)](https://github.com/apiad/starter/releases)
[![License](https://img.shields.io/github/license/apiad/starter?style=for-the-badge&color=success)](LICENSE)
[![Template](https://img.shields.io/badge/Repository-Template-blueviolet?style=for-the-badge&logo=github)](https://github.com/apiad/starter/generate)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg?style=for-the-badge)](https://github.com/apiad/starter/graphs/commit-activity)

**Transform how you work with AI agents.**

*A cognitive partnership model that enforces rigorous engineering standards, strategic planning, and continuous validation.*

#### 🚀 [**Check out the launch blog post!**](https://blog.apiad.net/p/how-im-using-ai-today)

</div>

---

## 🔥 Quick Start

The fastest way to bootstrap a new project or integrate the framework into an existing one is to run the following command in your terminal:

```bash
curl -fsSL https://apiad.github.io/starter/install.sh | bash
```

This interactive script will:
1.  **Validate** your environment (requires `git`, `node`, and a clean working tree).
2.  **Clone** the latest framework components to a temporary location.
3.  **Analyze** your current directory and propose a list of files to create or update.
4.  **Prompt** for your explicit confirmation before applying any changes.
5.  **Integrate** the `.gemini/` configuration and framework files (non-destructively).
6.  **Commit** the changes automatically with a descriptive message.

Once the installation is complete, run `gemini /onboard` to get an overview of the repository and start your first session.

### 🏗️ Alternative: Manual Setup

If you prefer a manual setup:

1. [Create a new project from this template](https://github.com/apiad/starter/generate).
2. Follow the onboarding instructions in the generated repository.

---

## 🧠 The Core Philosophy

This repository is a heavily customized and oppinionanted Gemini CLI agent, ready to be used in any situation. In this framework, the AI agent is not just a "code generator" or a "copilot". It is a **Senior Architect and Critical Thinking Partner**:

*   **🛡️ Critical Feedback First:** The agent is instructed to challenge unsafe, redundant, or poorly conceived ideas *before* writing a single line of code.
*   **📋 Research -> Plan -> Execute:** Every non-trivial change follows a strict lifecycle. The agent first researches context, proposes a detailed plan, waits for your explicit approval, and only then begins implementation.
*   **✅ Validation-Always:** The framework uses `make` as a source of truth. The agent is hooked into the `makefile` to ensure every change is validated (linted and tested) before being finalized.

## 🛠️ The Project Lifecycle

The `.gemini/commands/` directory defines specialized workflows that automate every phase of the development lifecycle:

### 🔍 Phase 1: Planning & Discovery
*   **`/research <topic>`**: A deep, 3-phase investigation (Planning -> Data Gathering -> Reporting) that produces exhaustive Markdown reports in the `research/` directory. **Crucial for gathering technical requirements and state-of-the-art context.**
*   **`/plan`**: The **Architectural Bridge**. This interactive workflow translates ideas into actionable execution plans:
    *   **Phase 1 (Clarification):** The agent interviews you to resolve ambiguities before planning.
    *   **Phase 2 (Agentic Analysis):** A specialized `planner` subagent scans the codebase and generates a detailed technical strategy.
    *   **Phase 3 (Artifact Generation):** A persistent Markdown plan is saved in `plans/` (e.g., `plans/feature-x.md`).
    *   **Phase 4 (Synchronization):** The plan is optionally linked to `TASKS.md` and can be synchronized with GitHub issues.
*   **`/onboard`**: Summarizes the project's architecture, standards, and current state to quickly orient a new developer (or the agent itself).

### 🏗️ Phase 2: Development & Execution
*   **`/issues`**: Your gateway to GitHub integration. It allows you to list, create, or update issues. Use `/issues work <number>` to transition an issue directly into a detailed research and planning mode.
*   **`/debug`**: Activates a specialized `debugger` subagent to perform forensic root-cause analysis (RCA). It analyzes error logs, traces code execution, and generates structured reports to pinpoint bugs.
*   **`/task`**: Manages the project roadmap in `TASKS.md`. Use it to `create` new tasks, `work` on existing ones (marks as In Progress), `report` on priorities, or `update` the roadmap.
*   **`/scaffold`**: Initializes new project structures from scratch using modern, standard tooling (Python/uv, TS/npm, Rust/cargo, etc.) and sets up a compatible `makefile`.

### 🧹 Phase 3: Content Generation, Maintenance & Documentation
*   **`/draft`**: Multi-phase workflow to turn research and plans into detailed, high-quality technical documents or articles section-by-section.
*   **`/revise`**: Uses the `editor` subagent to perform structural and linguistic audits based on the project's style guide.
*   **`/maintenance`**: Performs a deep scan of the codebase to identify technical debt, refactoring opportunities, and areas to improve test coverage.
*   **`/docs`**: Analyzes the codebase and journals to generate or update comprehensive project documentation in the `docs/` folder.
*   **`/cron`**: Manages repetitive background tasks (e.g., health checks, automated reports) using **systemd user timers**.

### 🚀 Phase 4: Shipping & Quality
*   **`/commit`**: Analyzes all uncommitted changes, groups them into logical features or fixes, and guides you through committing them individually using **Conventional Commits**.
*   **`/release`**: Automates the final shipping steps: running tests, determining the next version (semver), updating the `CHANGELOG.md`, and tagging the release.

## 🔄 Common Use Cases

This framework shines when you combine these commands into cohesive workflows:

### 1️⃣ Feature Development Workflow
1.  **Discover:** Run `/research` to understand the domain or library.
2.  **Plan:** Use `/plan` to turn requirements into a technical roadmap in `plans/`.
3.  **Track:** Link the plan to `TASKS.md` using `/plan`'s built-in sync.
4.  **Implement:** Use `/task work` to mark progress and begin coding.
5.  **Refine:** Run `/docs` to ensure your changes are well-documented.
6.  **Ship:** Use `/commit` for clean history and `/release` for a new version tag.

### 2️⃣ Bug Resolution Workflow
1.  **Triage:** Use `/issues summary` to see what needs attention.
2.  **Analyze:** Use `/debug` to activate the forensic subagent and perform root-cause analysis (RCA).
3.  **Fix:** Develop the fix based on the RCA and validate with `make`.
4.  **Sync:** Update the issue with the resolution details using `/issues`.

### 3️⃣ Content Creation Workflow
1.  **Research:** Use `/research` to gather data into an executive report.
2.  **Draft:** Run `/draft` to build an outline and expand it into a full article.
3.  **Revise:** Use `/revise` for step-by-step style guide-driven audits and refinements.

## ⚓ The Hook System

The framework uses a robust hook system (`.gemini/hooks/`) that synchronizes the agent with your project state:

*   **`session.py`**: Initializes the environment and provides a project summary.
*   **`journal.py`**: Ensures a journal entry exists for the current date (`journal/YYYY-MM-DD.md`).
*   **`make.py`**: Automatically runs `make` after critical agent actions to prevent regressions.
*   **`cron.py`**: Synchronizes `cron.toml` tasks with **systemd user timers**.

## 📄 License & Contribution

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. All contributions are welcome!
