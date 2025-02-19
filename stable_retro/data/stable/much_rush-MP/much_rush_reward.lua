current_score = 31
frame_penalty = -0.00005  -- Negative reward per frame

function reward()
    -- Initialize local reward
    local local_reward = 0

    -- Reward for score decrease
    if data.score < current_score then
        current_score = data.score
        local_reward = local_reward + 0.032
    end

    -- Add frame penalty
    local_reward = local_reward + frame_penalty

    return local_reward
end
