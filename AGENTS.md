# Agent Collaboration Rules — Nuri

Nuri is an encoder-only model project. These rules were established during the **design phase**, when multiple agents, including Claude and Codex, exchanged ideas through shared documents to develop the architecture.

## File Responsibilities

| File | Purpose | Editing rules |
|---|---|---|
| `docs/ARCHITECTURE.md` | The design's **single source of truth** | Make targeted edits only. **Do not rewrite the entire document.** Do not change user decisions D-001 through D-008 in the §14 decision log without user approval. |
| `docs/DISCUSSION.md` | Proposals, objections, and review threads | **Append only.** Do not edit another participant's entries. |
| `docs/design_review.md` | Historical record of the Codex subagents' joint v0 review | Keep it as a record. Add subsequent updates to DISCUSSION. |
| `docs/archive/` | Earlier versions, including Codex v0 and Claude v0.2 | Read only. |
| `configs/nuri_v0.design.json` | Architecture declaration, not an executable training configuration | Update it together with ARCHITECTURE.md. |
| `.dev/` | The user's original idea notes, excluded from Git | **Read only.** |
| `.private/` | Private notes, including data provenance, excluded from Git | Keep all data-related notes here. |

## Data Confidentiality Rule (D-008)

**Do not include training dataset names, links, or sources in tracked files or commit messages.** This includes documentation, configurations, code, and model cards. Record data-related information only in `.private/`. Code must obtain data paths from environment variables or a configuration under `.private/`.

## Workflow

1. Before starting work, read all of `docs/ARCHITECTURE.md`, especially the §14 decision log and §17 open questions, along with the latest entries in `docs/DISCUSSION.md`.
2. Post new ideas and objections in `DISCUSSION.md` first.
3. For small corrections, such as typos, formula errors, or clear omissions, update `ARCHITECTURE.md` directly and leave a one-line note in DISCUSSION.
4. Apply architecture changes to `ARCHITECTURE.md` after agreement in DISCUSSION or approval from the user. Record finalized decisions in §14.
5. When evidence helps answer an open question (`Q-*`), add your assessment to DISCUSSION. The user makes the final decision.

## Change Notifications (`tools/shared_watch.py`)

The watcher reports changes to shared files against each agent's snapshot: `AGENTS.md`, `CLAUDE.md`, `docs/**/*.md` except archives, and `configs/**/*.json`. Snapshots are stored in `.watch/<agent>.json`, which is excluded from Git.

| Mode | Command | Purpose |
|---|---|---|
| watch | `python tools/shared_watch.py watch --agent <name>` | Run continuously and print one diff summary for each change. Use this when an agent can receive process-output events, such as Claude Code's Monitor. |
| wait | `python tools/shared_watch.py wait --agent <name>` | Wait for the next change, print its diff, then exit. The default timeout is 1,800 seconds. Use this when an agent can run commands but cannot receive output events directly, such as Codex. |
| check | `python tools/shared_watch.py check --agent <name>` | Immediately print changes since the last check and update the snapshot. |
| ack | `python tools/shared_watch.py ack --agent <name>` | Update the snapshot without printing a diff. Run this immediately after your own edit so it does not trigger a notification back to you. |

Rules:

- **Claude:** Keep `watch --agent claude` running through Monitor during the session.
- **Codex:** Run `check --agent codex` when starting work and immediately before editing a shared file. When waiting after completing available work, run `wait --agent codex`. When it reports changes, read the original files, respond as needed, and run `wait` again. Restart `wait` after a timeout.
- Run `ack --agent <name>` immediately after editing a shared file.
- Notification diffs are summaries. Read the original content before making a decision.

## Concurrent Editing

- Before editing, append `🔒 <agent> editing <file> (start time)` to the end of `DISCUSSION.md`. After finishing, append `🔓 <agent> done <file>`. A lock left by another agent may be ignored if it has remained unreleased for more than 30 minutes.
- Use targeted edits instead of overwriting files. Read each file again before writing to check for changes made in the meantime.

## ID Rules

Assign each new entry the next number for its prefix. Never reuse an ID.

- `D-`: decisions; `Q-`: open questions, recorded in ARCHITECTURE.md §14 and §17.

## DISCUSSION Entry Format

```markdown
### [YYYY-MM-DD] <agent name> — <title>
- Scope: §n, D-xxx, Q-xxx
- Type: proposal | objection | review | question | response
- Details: ...
- Evidence: papers, calculations, or experimental results
- Status: open | accepted | rejected | merged
```

## Writing style

- Write README, docs, commit messages and code comments in **English**. Keep them easy to read and detailed.
- Keep code simple and intuitive. Prefer clear names and short functions over clever tricks.
- Commit one unit of work at a time (never batch unrelated changes). Each commit message has a summary line and a detailed body: what changed, why, and how it was tested.
- Commit only your own files with `git commit -- <paths>`, because the git index is shared between agents.
- Back claims with evidence (papers, FLOP counts, experiments). Mark numbers that were not measured as estimates.
- Earlier entries in `docs/DISCUSSION.md` were originally written in Korean. New entries must be in English; translate existing entries when the user requests it, preserving their meaning.
