
local max_values = {
    coins = 30,
    shroom = 7,
    egg = 9,
    flower = 5,
    cloud = 11,
    star = 3
}

function reward()
    local reward = 0  
    if data.coins > 30 then
        reward = 1
    end
    for key, max_value in pairs(max_values) do
        if key ~= "coins" then 
            if data[key] and data[key] < max_value then
                reward = reward + 0.01  
                max_values[key] = max_values[key] - 1  
            end
        end
    end

    return reward
end
