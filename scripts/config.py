CONFIG = {
    'broom_zoom': {
        'actions': [['UP'], ['DOWN'], ['UP', 'RIGHT'], ['DOWN', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'LEFT'], []],
        'game_env': 'broom_zoom-MP',
        'state': 'Level1',
        'clip_reward': True,
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
    },
    'amplifried': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], []],
        'game_env': 'amplifried-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (160, 144),
        'timesteps': 1000000,
    },
    'bunny_belt': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], []],
        'game_env': 'bunny_belt-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (160, 144),
        'timesteps': 1000000
    },
    'pest_aside': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], ['B']],
        'game_env': 'pest_aside-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (160, 144),
        'timesteps': 1000000000
    },
    'match-em': {
        'actions': [['UP'], ['DOWN'], ['A']],
        'game_env': 'match-em-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (160, 144),
        'timesteps': 1000000
    },
        'hammergeddon': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], ['B'], ['UP', 'B'], []],
        'game_env': 'hammergeddon-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 10000000
    }
}
