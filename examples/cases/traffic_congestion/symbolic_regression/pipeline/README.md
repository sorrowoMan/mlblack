# pipeline

- Responsibility: Data pipeline layer: data views, numericizer, feature engineering.
- Boundary: keep only this layer's concern; avoid cross-layer logic here.
- Context contract (if any):
  - context_requires / context_provides / context_mutates / context_cache
- Minimal example: keep one runnable file, or document the entry path.
