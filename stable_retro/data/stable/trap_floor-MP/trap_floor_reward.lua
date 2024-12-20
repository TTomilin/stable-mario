current_score = 42
reward = 0

function reward()
    local reward = -0.0001
    if data.score < current_score then
        current_score = current_score - 1
        reward = reward + 1
    end
    return reward
end
