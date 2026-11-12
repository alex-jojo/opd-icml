"""Read W&B settings from the command environment; never persist API keys."""


def load_wandb_environment(environment):
    env = environment.copy()
    env.setdefault('WANDB_MODE', 'online' if env.get('WANDB_API_KEY') else 'disabled')
    if env['WANDB_MODE'] not in {'online', 'offline', 'disabled'}:
        raise ValueError('WANDB_MODE must be online, offline, or disabled')
    if env['WANDB_MODE'] != 'disabled':
        env.setdefault('WANDB_PROJECT', 'opd-icml')
    return env


def apply_wandb_tracking(config, environment):
    """Return public Hydra overrides; credentials stay in the child environment."""
    trainer = config['trainer']
    original = trainer.get('logger', ['console'])
    loggers = [original] if isinstance(original, str) else list(original)
    if environment.get('WANDB_MODE', 'disabled') == 'disabled':
        loggers = [name for name in loggers if name not in ('wandb', 'tracking')]
        if not loggers:
            loggers = ['console']
    elif not any(name in loggers for name in ('wandb', 'tracking')):
        loggers.append('wandb')
    overrides = {}
    if loggers != original:
        trainer['logger'] = loggers
        overrides['trainer.logger'] = loggers
    project = environment.get('WANDB_PROJECT')
    if environment.get('WANDB_MODE') != 'disabled' and project and project != trainer.get('project_name'):
        trainer['project_name'] = project
        overrides['trainer.project_name'] = project
    return overrides
