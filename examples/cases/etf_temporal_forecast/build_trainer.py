from pathlib import Path
from mlblack.assembly.schema import load_scaffold_config
from mlblack.assembly import build_pipeline, build_trainer

def build_project_trainer(data, config_path=None):
    path = Path(config_path or Path(__file__).parent / 'config' / 'scaffold.json')
    config = load_scaffold_config(path)
    inner_training = dict(config.inner_training)
    pipeline = build_pipeline(inner_training.get('pipeline'))
    resource_context = dict(inner_training.get('resource_context', {}) or {})
    prepared = pipeline.fit_transform(data, resource_context)
    trainer_spec = dict(inner_training.get('trainer', {}) or {})
    if resource_context and not trainer_spec.get('resource_context'):
        trainer_spec['resource_context'] = resource_context
    trainer = build_trainer(trainer_spec, prepared)
    trainer.context_store['pipeline'] = pipeline.describe()
    return trainer
