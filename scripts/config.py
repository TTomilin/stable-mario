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
        'resize': (80, 72),
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
    }
}
