current_location = 92
i = 0

function reward()
    local reward = -0.0001
    if data.location < current_location then
        current_location = current_location - 1
        reward = reward + 1
        i = i+1
    end
    if data.location > current_location then
        reward = -i/2
    end
    return reward
end
