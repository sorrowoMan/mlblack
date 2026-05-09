# Family Router Protocol

`mlblack` treats a family preset as a formal routing surface, not just another algorithm name.

Current formal family presets:

- `linear`
- `neural`
- `tree_boosting`
- `symbolic`

## Router Contract

Each formal family preset should expose:

- `formal_preset`
- `route_registry`
- `route_keys`
- `surface_status`

Each route should expose:

- `family_key`
- `route_key`
- `match_fields`
- `status`
- `summary`

Runtime trainers assembled from a family route should expose:

- `family_router_family`
- `family_router_target`
- `family_route_spec`
- `family_route_registry`

Family-specific aliases may also be attached, for example:

- `linear_router_target`
- `neural_router_target`
- `tree_boosting_router_target`
- `symbolic_router_target`

## Routing Rule

The family surface should decide:

1. which fields are legal router inputs
2. how those fields match a concrete route
3. how conflicts are reported
4. how unsupported combinations are reported
5. how route contracts are projected to catalog fields

## Catalog Projection

Family and preset catalog entries should expose:

- `family_route_count`
- `family_route_keys`
- `family_route_match_fields`
- `family_route_statuses`
- `family_route_formal_preset`

Symbolic currently also keeps legacy compatibility fields such as:

- `symbolic_route_keys`
- `symbolic_route_backends`
- `symbolic_route_tasks`
- `symbolic_route_structure_modes`

## Component / Provider / Plugin Mount Contract

These three kinds should share one contract vocabulary in catalog:

- `mount_plane`
- `mount_point`
- `orchestration_phases`
- `contract_consumes`
- `contract_provides`
- `contract_mutates`

Additional kind-specific fields may still exist:

- component: `binding_level`, `signal_names`, `provides_fields`
- provider: `plane`, `supports_batch`, `supports_individual`
- plugin: `hook_events`, `contract_cache`, `priority`

## Design Intent

This protocol keeps `mlblack` closer to a framework-style assembly model:

- family defines the main fitting skeleton
- router decides the concrete preset target
- component/provider/plugin attach through explicit mount contracts
- catalog and UI can inspect the same structure without hard-coded special cases
