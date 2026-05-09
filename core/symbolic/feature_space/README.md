This package holds reusable symbolic feature-space components that are not tied to a specific dataset or optimization scaffold.

Current shared components:
- `activation_config.py`: dynamic family-budget and grammar activation settings
- `temporal_feature_pack.py`: lag-derived rolling, momentum, cross, and ratio features
- `regime_feature_pack.py`: volatility, shock, and regime-state lag features

Planned next extraction targets from scenario code:
- primitive registry / DSL primitive families
- grammar-based recursive feature generation
- candidate pool orchestration

Rule of thumb:
- Put generic feature-space logic here
- Keep dataset-specific routing, strict4 policies, and experiment assembly in scenario packages
