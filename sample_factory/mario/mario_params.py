def mario_override_defaults(_env, parser):
    """RL params specific to Mario envs."""
    parser.set_defaults(
        encoder_conv_architecture="convnet_atari",
        obs_scale=255.0,
        env_frameskip=4,
        env_framestack=4,
        exploration_loss_coeff=0.01,
        rollout=128,
        num_epochs=4,
        num_batches_per_epoch=4,
    )
