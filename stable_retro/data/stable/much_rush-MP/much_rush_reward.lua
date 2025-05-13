current_score = 31
num_lives = 3
function reward()
    local_reward = 0
    if data.score < current_score then
        current_score = data.score
        local_reward = local_reward + 0.032
    end
    if data.lives < num_lives then
        num_lives = data.lives
        local_reward = local_reward - 0.33
    end
    return local_reward
end
