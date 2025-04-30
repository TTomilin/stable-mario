previous_score = 10


function reward()
    local score_rew = - data.score + previous_score -0.0001
    previous_score = data.score
    return score_rew / 10
end