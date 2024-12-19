koopastart = 47

function reward()
    local reward = koopastart - data.koopa_ice - 0.001
    koopastart = data.koopa_ice
    return reward
end