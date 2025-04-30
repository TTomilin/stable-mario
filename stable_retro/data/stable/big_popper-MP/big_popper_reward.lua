prev_best_score = 0

function reward ()
    local reward = 0

    local current_score = data.score
    if current_score > prev_best_score then
        reward = reward + current_score - prev_best_score
        prev_best_score = current_score
    end

    return reward / 86
end