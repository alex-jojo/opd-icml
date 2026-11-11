"""Shared W&B credentials for every training entry point; never include keys in Hydra."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRIVATE_CONFIG = ROOT / '.wandb.local.json'
FIELDS = {'WANDB_API_KEY', 'WANDB_MODE', 'WANDB_PROJECT', 'WANDB_ENTITY'}


def load_wandb_environment(environment):
    env = environment.copy()
    path = Path(env.get('OPD_WANDB_CONFIG', PRIVATE_CONFIG)).expanduser()
    if path.is_file():
        try:
            settings = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            raise ValueError('Cannot read private W&B configuration; expected a JSON object') from None
        if not isinstance(settings, dict) or any(
                key not in FIELDS or not isinstance(value, str) for key, value in settings.items()):
            raise ValueError('Private W&B configuration must contain only supported WANDB string settings')
        for key, value in settings.items():
            if value and not env.get(key):
                env[key] = value
    elif 'OPD_WANDB_CONFIG' in env:
        raise ValueError('OPD_WANDB_CONFIG does not point to a private configuration file')
    env.setdefault('WANDB_MODE', 'online' if env.get('WANDB_API_KEY') else 'offline')
    return env


def apply_wandb_tracking(config, environment):
    """Enable the existing W&B backend; return only non-secret Hydra overrides."""
    if environment.get('WANDB_MODE') == 'disabled':
        return {}
    if not environment.get('WANDB_API_KEY') and environment.get('WANDB_MODE') != 'online':
        return {}
    trainer = config['trainer']
    loggers = trainer.get('logger', ['console'])
    loggers = [loggers] if isinstance(loggers, str) else list(loggers)
    overrides = {}
    if not any(name in loggers for name in ('wandb', 'tracking')):
        loggers.append('wandb')
        trainer['logger'] = loggers
        overrides['trainer.logger'] = loggers
    project = environment.get('WANDB_PROJECT')
    if project and project != trainer.get('project_name'):
        trainer['project_name'] = project
        overrides['trainer.project_name'] = project
    return overrides
