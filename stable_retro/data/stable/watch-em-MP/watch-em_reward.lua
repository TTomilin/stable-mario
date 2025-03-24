current_coins = 30

function reward()
    local reward = 0
    if data.coins > current_coins then 
        reward = (data.coins - current_coins) / 10
        current_coins = data.coins
    elseif data.ingame == 1 then
        reward = -0.0001
        current_coins = data.coins
    end
    return reward
end

