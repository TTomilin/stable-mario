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
        'state': 'Level2',
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
        'actions': [['LEFT'], ['RIGHT'], ['A'], ['B'], ['A', 'B'], ['RIGHT','A'], ['LEFT','A'],['RIGHT','B'], ['LEFT','B']],
        'game_env': 'hammergeddon-MP',
        'state': 'Level1',
        'clip_reward': False,
        'resize': (80, 72),
        'timesteps': 10000000
    },
        'sort_stack': {
        'actions': [['UP'], ['DOWN'], ['LEFT'], ['RIGHT'], ['A'], ['B'], []],
        'game_env': 'sort_stack-MP',
        'state': 'Sort_Stack',
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
        },
        'shell_stack':{
            'actions': [[], ['A']],
            'game_env': 'shell_stack-MP',
            'state': 'shell_stack',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'koopa_crunch':{
            'actions': [[], ['A'],['B'],['DOWN'],['LEFT'],['RIGHT']],
            'game_env': 'koopa_crunch-MP',
            'state': 'koopa_crunch',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'stop_em':{
            'actions': [[], ['A'],['DOWN'],['UP']],
            'game_env': 'stop_em-MP',
            'state': 'stop_em',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'sled_slide':{
            'actions': [[], ['A'],['B'],['A','LEFT'],['LEFT'],['RIGHT'],['A','RIGHT'],['B','LEFT'],['B','RIGHT']],
            'game_env': 'sled_slide-MP',
            'state': 'sled_slide',
            'clip_reward': False,
            'resize': (40, 40),
            'timesteps': 129740258283
        },
        'barrel_peril':{
            'actions': [['B', 'LEFT'], ['UP', 'B'], ['RIGHT', 'B'], ['DOWN', 'B'],['A'],['B', 'LEFT','UP'], ['UP', 'B','RIGHT'], ['RIGHT', 'B','DOWN'], ['DOWN', 'B','LEFT']],
            'game_env': 'barrel_peril-MP',
            'state': 'barrel_peril',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 10000000000
        },
        'boo_bye':{
            'actions': [[],['A'],['LEFT'],['RIGHT'],['A','LEFT'],['A','RIGHT']],
            'game_env': 'boo_bye-MP',
            'state': 'boo_bye',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 10000000000
        },
        'crushed_ice':{
            'actions': [[], ['LEFT'],['DOWN'],['UP'],['RIGHT'],['RIGHT','UP'],['RIGHT','DOWN'],['LEFT','UP'],['LEFT','DOWN']],
            'game_env': 'crushed_ice-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'floor_it':{
            'actions': [[], ['DOWN'],['UP'],['A'],['B'],['A','B']],
            'game_env': 'floor_it-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'splatterball':{
            'actions': [[], ['LEFT'],['DOWN'],['UP'],['RIGHT'],['RIGHT','UP'],['RIGHT','DOWN'],['LEFT','UP'],['LEFT','DOWN'],['A'], ['A','LEFT'],['A','DOWN'],['A','UP'],['A','RIGHT'],['A','RIGHT','UP'],['A','RIGHT','DOWN'],['A','LEFT','UP'],['A','LEFT','DOWN'],['R'], ['R','LEFT'],['R','DOWN'],['R','UP'],['R','RIGHT'],['R','RIGHT','UP'],['R','RIGHT','DOWN'],['R','LEFT','UP'],['R','LEFT','DOWN'],['B'], ['B','LEFT'],['B','DOWN'],['B','UP'],['B','RIGHT'],['B','RIGHT','UP'],['B','RIGHT','DOWN'],['B','LEFT','UP'],['B','LEFT','DOWN'],['B','A'], ['B','A','LEFT'],['B','A','DOWN'],['B','A','UP'],['B','A','RIGHT'],['B','A','RIGHT','UP'],['B','A','RIGHT','DOWN'],['B','A','LEFT','UP'],['B','A','LEFT','DOWN'],['B','R'], ['B','R','LEFT'],['B','R','DOWN'],['B','R','UP'],['B','R','RIGHT'],['B','R','RIGHT','UP'],['B','R','RIGHT','DOWN'],['B','R','LEFT','UP'],['B','R','LEFT','DOWN']],
            'game_env': 'splatterball-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'go_go_pogo':{
            'actions': [[], ['LEFT'],['RIGHT'],['A'],['LEFT','A'],['A','RIGHT']],
            'game_env': 'go_go_pogo-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'bob_ooom':{
            'actions': [[], ['LEFT','B'],['RIGHT','B'],['A']],
            'game_env': 'bob_ooom-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'reel_cheep':{
            'actions': [[], ['LEFT'],['RIGHT'],['UP'],['DOWN'],['LEFT','UP'],['RIGHT','UP'],['LEFT','DOWN'],['RIGHT','DOWN'],['L'],['R']],
            'game_env': 'reel_cheep-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'see_monkey':{
            'actions': [[], ['LEFT','A'],['RIGHT','A'],['A'],['LEFT'],['RIGHT']],
            'game_env': 'see_monkey-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'tankdown':{
            'actions': [[], ['LEFT','A'],['RIGHT','A'],['A'],['LEFT'],['RIGHT'],['UP'],['UP','A'],['DOWN'],['DOWN','A']],
            'game_env': 'tankdown-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'chicken':{
            'actions': [[], ['A'],['B']],
            'game_env': 'chicken-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'outta_my_way':{
            'actions': [[], ['A'],['B'],['LEFT'],['RIGHT'],['A','LEFT'],['A','RIGHT'],['B','LEFT'],['B','RIGHT']],
            'game_env': 'outta_my_way-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'dreadmill':{
            'actions': [[],['LEFT'],['RIGHT'],['UP','LEFT'],['UP','RIGHT'],['DOWN','LEFT'],['DOWN','RIGHT'],['DOWN'],['UP']],
            'game_env': 'dreadmill-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'chomp_walker':{
            'actions': [[],['LEFT'],['RIGHT'],['UP','LEFT'],['UP','RIGHT'],['DOWN','LEFT'],['DOWN','RIGHT'],['DOWN'],['UP']],
            'game_env': 'chomp_walker-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
	    'switch_way':{
            'actions': [[], ['A'],['LEFT'],['RIGHT'],['A','LEFT'],['A','RIGHT']],
            'game_env': 'switch_way-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'trap_floor':{
            'actions': [[],['A'],['B'],['A','B']],
            'game_env': 'trap_floor-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'stair_scare':{
            'actions': [[],['A']],
            'game_env': 'stair_scare-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'chainsaw':{
            'actions': [[],['L'],['R'],['L','R']],
            'game_env': 'chainsaw-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'scratch-em':{
            'actions': [[], ['A'], ['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['A', 'LEFT'], ['A', 'RIGHT'], ['A', 'UP'],['A', 'DOWN']],
            'game_env': 'scratch-em-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'pair-em':{
            'actions': [[], ['A'], ['LEFT'], ['RIGHT'], ['UP'], ['DOWN'], ['A', 'LEFT'], ['A', 'RIGHT'], ['A', 'UP'],['A', 'DOWN']],
            'game_env': 'pair-em-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'koopa_kurl':{
            'actions': [[], ['A'], ['LEFT'], ['RIGHT']],
            'game_env': 'koopa_kurl-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        'watch-em':{
            'actions': [[], ['A'], ['LEFT'], ['RIGHT'],['UP']],
            'game_env': 'watch-em-MP',
            'state': 'Level1',
            'clip_reward': False,
            'resize': (160, 160),
            'timesteps': 129740258283
        },
        
}
