current_location = 92
count = 0
function reward()
    local reward = 0
    if data.ingame == 0 and current_location ~= 1000 then
        reward = -count + 50 - count / 10
    elseif current_location == 1000 then
        reward = 0
    else
        if data.location < current_location then
            current_location = data.location
            reward = reward + 0.25
            count = count + reward
        end
        elseif data.location > current_location then
            reward = -count / 2
            current_location = 1000
        end
    end
    return reward / 45
end