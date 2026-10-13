Show which skills are available, so they stop being invisible. Skills are loaded by the model through
the `skill` tool when a task matches; they are not `/` commands, and this command exists because that
makes them hard to discover.

## Usage

```
/skills
```

## Instructions

1. List the project skills from `.opencode/skills/*/SKILL.md`, printing each directory name with the
   one-line description from its frontmatter.
2. List the global skills from `~/.config/opencode/skills/` the same way.
3. Group the project set by what it is for: this repo's own (`cpp-cuda-review`, `how`,
   `figure-it-out`, `ncu-report`, `cuda-debugging`, `address-sanitizer`, `sanitizers`,
   `reflect`, `show-me-your-work`, `opportunity-scan`, `c-review`, `fp-check`), the
   language-agnostic pair (`desloppify`, `convention-learner`), and the `principle-*` set.
   The .NET/Roslyn skills that used to sit beside them were removed on 2026-10-13; see
   `.opencode/AGENTS.md` for the check that justified it.
4. Note two things the user cannot see from the UI: skills are invoked by asking for them (for
   example "review this with cpp-cuda-review"), and subagents are invoked with `@`, not `/`.
