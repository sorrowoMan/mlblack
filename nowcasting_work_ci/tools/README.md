# tools

Operational helpers that are not part of the formal runtime entry live here.

Current examples:

- deterministic smoke regression
- result aggregation and plotting

These tools may be re-exported by top-level shims for compatibility, but their
implementation should stay under `tools/`.
