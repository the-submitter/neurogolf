# Add max and ultra Codex reasoning efforts

Status: independently reviewed and explicitly authorized by the maintainer on
2026-07-14. Implemented with an intentional integrity-manifest refresh.

## Requested change

Update `CodexConfig.reasoning_effort` in `neurogolf/config/models.py` from:

```python
Literal["low", "medium", "high", "xhigh"] = "xhigh"
```

to:

```python
Literal["low", "medium", "high", "xhigh", "max", "ultra"] = "max"
```

This makes the typed fallback configuration accept every reasoning effort that
the installed Codex 0.144.3 catalog reports for `gpt-5.6-sol`, and makes `max`
the repository fallback default. The normal NeuroGolf invocation continues to
select `max` through `~/.codex/neurogolf-max.config.toml`.

## Reproduction and evidence

1. Run `codex debug models --bundled` and inspect `gpt-5.6-sol`.
2. Confirm its supported efforts are `low`, `medium`, `high`, `xhigh`, `max`,
   and `ultra`.
3. Construct `CodexConfig(reasoning_effort="max")` or
   `CodexConfig(reasoning_effort="ultra")` with the current repository model.
4. Observe that Pydantic rejects both values because the protected model only
   permits efforts through `xhigh`.

## Scope and parity impact

This is an orchestrator configuration expansion. It does not change canonical
task data, ARC-GEN generators, ONNX validation, scorer conversion, gate
semantics, sealed tests, champion promotion, or competition parity. No task is
affected unless a maintainer explicitly selects one of the new reasoning
efforts for a Codex worker.

## Required tests

- Assert `CodexConfig()` defaults to `reasoning_effort == "max"`.
- Assert `max` and `ultra` validate successfully.
- Assert unsupported strings remain rejected.
- Update the no-profile command-builder test to expect
  `model_reasoning_effort="max"`.
- Re-run the complete unit, integration, and parity suite plus Ruff and Pyright.

## Integrity migration

The current protected file hash is
`00b06f645b43d1c8a1f96398a64fe94e324f8bf7a1aa2b981ea86ca6ce39e994`.
Applying only the requested enum/default line changes it to
`e8f1fae6078d684a7e0f88bb9d26db023cd5a80b3d24a66625c4093574b2373d`.
Do not update `configs/integrity.json` until an independent review using
`prompts/review_test_change.md` recommends the change and a maintainer performs
an intentional manifest refresh.
