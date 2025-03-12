current_coins = 30

function reward()
    local reward = 0
    if data.coins > current_coins then 
        reward = data.coins - current_coins
        current_coins = data.coins
    end
    return reward
end

