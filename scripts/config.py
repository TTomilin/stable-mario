CONFIG = {
    'broom_zoom': {
        'actions': [['UP'], ['DOWN'], ['UP', 'RIGHT'], ['DOWN', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'LEFT'], []],
        'game_env': 'broom_zoom-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (40, 36),
        'timesteps': 100000000000,
    },
    'spook_spike': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], []],
        'game_env': 'spook_spike-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 1000000,
    },
    'flippin_out': {
        'actions': [['UP', 'A'], ['DOWN', 'A'], ['LEFT', 'A'], ['RIGHT', 'A'], ['RIGHT', 'UP', 'A'], ['RIGHT', 'DOWN', 'A'], ['LEFT', 'UP', 'A'], ['LEFT', 'DOWN', 'A'], []],
        'game_env': 'flippin_out-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 100000000000,
    },
    'on_the_spot': {
        'actions': [['LEFT'], ['RIGHT'], ['UP'], ['DOWN']],
        'game_env': 'on_the_spot-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (40, 36),
        'timesteps': 100000000000,
    },
    'amplifried': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], []],
        'game_env': 'amplifried-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 1000000000,
    },
    'bill_bounce': {
        'actions': [['LEFT'], ['RIGHT'], ['LEFT', 'A'], ['RIGHT', 'A'], ['A'], []],
        'game_env': 'bill_bounce-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 1000000000
    },
    'bunny_belt': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A']],
        'game_env': 'bunny_belt-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (40, 36),
        'timesteps': 1000000000000
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
    },
        'sort_stack': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], ['B'], []],
        'game_env': 'sort_stack-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (160, 144),
        'timesteps': 10000000
    },
        'stompbot_xl': {
        #'actions': [['UP'], ['UP', 'A'], ['UP', 'B'], ['LEFT'], ['LEFT', 'A'], ['LEFT', 'B'], ['RIGHT'], ['RIGHT', 'A'], ['RIGHT', 'B'], ['A'], ['B'], ['UP', 'RIGHT'], ['UP', 'RIGHT', 'A'], ['UP', 'RIGHT', 'B'], ['UP', 'LEFT'], ['UP', 'LEFT', 'A'], ['UP', 'LEFT', 'B'], []],
        'actions': [['UP', 'A'], ['LEFT', 'A'], ['RIGHT', 'A'], ['UP', 'RIGHT', 'A'], ['UP', 'LEFT', 'A'], ['A']],
        'game_env': 'stompbot_xl-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 10000000000
    },
        'fling_shot': {
        'actions': [['B'], ['A'], ['UP', 'RIGHT'], ['UP', 'LEFT'], ['DOWN', 'RIGHT'], ['DOWN', 'LEFT'], ['UP'], ['DOWN'], ['LEFT'], ['RIGHT']],
        'game_env': 'fling_shot-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 10000000000
    },
        'big_popper': {
        'actions': [['A'], ['LEFT'], ['RIGHT'], ['LEFT', 'A'], ['RIGHT', 'A']],
        'game_env': 'big_popper-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 10000000000
    },
        'melon_folly': {
        'actions': [['B', 'LEFT'], ['UP', 'B'], ['RIGHT', 'B'], ['DOWN', 'B']],
        'game_env': 'melon_folly-MP',
        'state': '',
        'clip_reward': False,
        'timesteps': 10000000000
    },
        'cloud_climb': {
        'actions': [['A'], ['LEFT'], ['RIGHT'], ['LEFT', 'A'], ['RIGHT', 'A'], []],
        'game_env': 'cloud_climb-MP',
        'state': 'cloud_climb',
        'clip_reward': False,
        'resize': (60,40),
        'timesteps': 10000000000
    },
        'grabbit': {
        'actions': [['B', 'LEFT'], ['B', 'A', 'LEFT'], ['B', 'RIGHT'], ['RIGHT', 'A', 'B'], ['B', 'UP'], ['UP', 'A', 'B'], ['B', 'DOWN'], ['DOWN', 'A', 'B'], ['B', 'LEFT', 'UP'], ['B', 'A', 'LEFT', 'UP'], ['B', 'RIGHT', 'UP'], ['RIGHT', 'A', 'B', 'UP'], ['B', 'LEFT', 'DOWN'], ['B', 'A', 'LEFT', 'DOWN'], ['B', 'RIGHT', 'DOWN'], ['RIGHT', 'A', 'B', 'DOWN']],
        'game_env': 'grabbit-MP',
        'state': 'grabbit',
        'clip_reward': False,
        'resize': (160,240),
        'timesteps': 10000000000
        },
        'forest_jump': {
            'actions': [[], ['A'], ['DOWN'],['LEFT'], ['RIGHT'], ['A','RIGHT'], ['A', 'LEFT']],
            'game_env': 'forest_jump-MP',
            'state': 'forest_jump',
            'clip_reward': False,
            'resize': (80, 120),
            'timesteps': 1927420971
        },
        'drop_em': {
            'actions': [[], ['A']],
            'game_env': 'drop_em-MP',
            'state': 'drop_em',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        }
}
