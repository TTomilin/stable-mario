current_coins = 30

function reward()
    local reward = 0
    if data.coins > current_coins then
        reward = 1
        current_coins = data.coins
    end
    return reward
end
