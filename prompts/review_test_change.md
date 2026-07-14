# Review a proposed test or judge change

Treat the external gate and canonical data as protected. Read every applicable
`AGENTS.md` and the request under `test_change_requests/`, reproduce the claimed
defect without modifying protected files, and check whether the proposal corrects
competition parity or merely weakens acceptance. Work only in the assigned review
scope. Report evidence, affected tasks, scorer parity impact, required
migration/hash refresh, and a recommendation. Do not merge or refresh integrity
hashes as part of this review. Write a schema-valid `state/codex_result.json`.
