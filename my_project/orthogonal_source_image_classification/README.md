# orthogonal_source_image_classification

Standard scaffold for probing the Orthogonal Source Layer on image classification data.

This is not a CNN benchmark. It tests whether source governance helps fixed
classification families after image observations have first been converted into
representation/source-object candidates.

Layer order:

`raw pixels -> searchable symbolic phi pool -> selected representation objects -> orthogonal source governance -> downstream classification family`

The important boundary is that flattened pixels are treated as raw observations,
not as source objects. Orthogonalization is applied to stroke, patch, edge, and
mass-distribution representation features.

This scaffold now exposes the objectification formulas as auditable candidates.
The local selection policy is intentionally simple: train-set class signal plus a
pairwise redundancy cap. A full `nsgablack` outer solver can replace that policy
without changing the downstream `mlblack` evaluation proxy.

The proxy accepts typed PhiBundle lane parameters from an outer solver. Examples:
edge direction/scope/operator, patch size/stride/pooling, texture operator,
DCT band/orientation, moment axis/statistic, and spatial projection bands.

Current dataset:

- `digits`: sklearn handwritten digits, 8x8 grayscale images, 10 classes.

Feature spaces:

- `raw_pixels`: flattened pixel reference baseline.
- `formula_pool`: all generated symbolic objectification formulas before selection.
- `image_representation`: selected stroke/patch/edge/frequency formulas before orthogonal governance.
- `orthogonal_sources`: selected class-aware source objects from `image_representation`.
- `image_representation_plus_orthogonal_sources`: representation features augmented with selected sources.

For this image scaffold, orthogonal source governance runs in identity-selection
mode. The searched `phi` layer owns formula construction; the source layer then
audits redundancy and survival rather than generating a second formula expansion
over the representation objects.

Audits:

- `representation_formula_table.csv`: candidate phi formulas, scores, selected flag, and formula family.
- `orthogonal_source_table.csv`: source-object candidates selected after representation objectification.

Run:

```powershell
python my_project\orthogonal_source_image_classification\run_solver.py --check
python my_project\orthogonal_source_image_classification\run_solver.py --suite-id digits_v1
```
