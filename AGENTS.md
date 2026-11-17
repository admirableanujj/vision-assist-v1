# AGENTS.md

Instructions for any AI coding agent working in this repository (Codex,
Claude Code, Gemini CLI, Antigravity, Cursor, Copilot, Grok, or anything
else that reads this file).

## Repository guidance

`CLAUDE.md` holds the repository layout, request flow, commands, and key
design decisions. Despite its name, nothing in it is specific to one agent:
read it before making changes.

## Skills

Agent skills live in `.agents/skills/`, in the open Agent Skills format (a
folder with a `SKILL.md` whose frontmatter has a `name` and a `description`).
Harnesses that support the format discover them there automatically.
`.claude/skills/` contains only symlinks into `.agents/skills/`, so there is
one copy of every skill. Add new skills under `.agents/skills/` and never
edit through the symlink path.

If your harness has no skill support, treat the `SKILL.md` as a procedure:
when a task matches a skill below, read that file in full and follow it.

| Task | Skill |
|---|---|
| Fine-tune YOLO on Kaggle, run or compare a hyperparameter sweep, or set up Kaggle credentials for the fine-tuning pipeline | `.agents/skills/kaggle-yolo-finetune/SKILL.md` |

The skills are interactive. Where one says to ask the user or to confirm
before an action, do that in whatever way your harness talks to the user;
do not skip the question because no dedicated question tool exists.
