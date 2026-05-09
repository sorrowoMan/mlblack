# Experiment Run Contract

`mlblack` uses the shared runtime surface contract version:

- `run-surface.v1`

Formal code module:

- `experiment.contracts`

## 1. Records

The runtime catalog contract is composed of four records:

1. `SurfaceRecord`
2. `AssemblyRecord`
3. `ArtifactRecord`
4. `RunRecord`

These records are stored as JSON projections in the experiment tracker tables.

## 2. Why This Exists

The runtime catalog should not answer only:

- "Which trainer ran?"

It should answer:

- which scaffold surface ran
- what assembly stack was actually mounted
- what artifact was produced
- what result the run ended with

## 3. mlblack Mapping

In `mlblack`, the recommended mapping is:

- `surface_kind = flow`
- `driver_ref = trainer:<name>`
- `family_ref = family:<name>`
- `preset_ref = preset:<name>`
- `head_ref = head:<task>`

The run comparison surface is therefore not only `trainer_name`.

Primary comparison keys:

- `surface_signature`
- `assembly_signature`
- `subject_signature`
- `param_signature`

## 4. Tracker Projection

`ExperimentTrackerCapability` persists:

- `surface_record_json`
- `assembly_record_json`
- `run_record_json`
- `artifact_record_json`

Current table mapping:

- `experiment_run_catalog`
  - `surface_record_json`
  - `assembly_record_json`
  - `run_record_json`
- `experiment_artifact_catalog`
  - `artifact_record_json`

This keeps the first version of the contract database-visible without forcing
every field to become its own SQL column immediately.

## 5. Current Scope

The contract is intentionally ahead of the current UI.

That means:

- some fields are already populated today
- some fields are optional placeholders for future scaffold-aware surfaces

This is acceptable as long as:

- field keys stay stable
- signatures stay stable
- future population fills existing fields instead of inventing new synonyms

