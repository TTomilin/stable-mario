CONFIG = {
    'broom_zoom': {
        'actions': [['UP'], ['DOWN'], ['UP', 'RIGHT'], ['DOWN', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'LEFT'], []],
        'game_env': 'broom_zoom-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 1000000,
    },
    'spook_spike': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], []],
        'game_env': 'spook_spike-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 1000000,
    }
}
