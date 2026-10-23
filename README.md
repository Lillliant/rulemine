# RuleMine

Local-first CLI that mines coding-agent transcripts for **corrections**, **decisions**, **dead ends**, and **cross-workspace skill needs**, then drafts:

- `AGENTS.md` rules (recurring steers)
- `DECISIONS.md` (ADR-lite)
- portable `SKILL.md` stubs (reusable across repos)

Powered by **Ollama + Gemma** on-device. Human approval before any write.

> Your agents forget everything. This remembers it for them, and for you.

## Status

Scaffold / handoff stage. **Implementers: start at [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md).** Project rules: [AGENTS.md](./AGENTS.md).

Hacktoberfest Week 3 (Slow and Steady) -- due **2026-10-25 11:59 PM PDT**. Target: Best Use of Gemma.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) with `gemma2:2b` (already pulled on the host machine)

```bash
ollama list   # expect gemma2:2b
```

## Intended CLI (after implementation)

```bash
rulemine mine --since 7d --model gemma2:2b
rulemine status
rulemine review
rulemine write --rules --decisions --skills --dry-run
```

## Privacy

Transcripts can contain private code, paths, and secrets. RuleMine redacts before inference and keeps extraction on localhost. That is why open weights are the right design choice, not a nice-to-have.

## License

TBD (MIT intended for the challenge repo).
