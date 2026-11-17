# vLLM issue tracker

## Snapshot of `good first issue` (2026-11-15)
14 open, 11 already assigned, and most are deep refactors. None suits a first PR from this Mac.
The only unassigned ones:
- #28498 [Bug][RL] Port Conflict: unassigned, 23 comments. Read the thread first; it may be stale.
- #25750 Whisper feature tracking: an umbrella issue; individual sub-items might be small.
- #23957 Phi4Flash in V1: model port, needs a GPU to verify. Skip for now.
- #21948 Add tests for parallel sampling combos (assigned, but test-writing is a good *type* of first task).

## Better searches for a first PR
```bash
gh issue list -R vllm-project/vllm --label documentation --state open --limit 30
gh issue list -R vllm-project/vllm --label bug --search "no:assignee CPU in:title" --limit 30
gh issue list -R vllm-project/vllm --search "no:assignee frontend OR openai in:title sort:created-desc" --limit 30
```
Typo/doc PRs to `docs/` are welcome and teach you the PR process.

## Candidates
| # | Title | Why it fits | Status |
|---|---|---|---|
| | | | |
