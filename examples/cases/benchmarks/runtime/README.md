# runtime

- Responsibility: L0 runtime: resources, compute pool, stores, artifact transport.
- Boundary: keep only this layer's concern; avoid cross-layer logic here.
- Context contract (if any):
  - context_requires / context_provides / context_mutates / context_cache
- Minimal example: keep one runnable file, or document the entry path.
