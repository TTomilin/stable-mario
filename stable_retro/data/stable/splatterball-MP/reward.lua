lastscore = 50
function reward()
    local reward = lastscore - data.score - 0.001
    lastscore = data.score
    return reward / 43
end