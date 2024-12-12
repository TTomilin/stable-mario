x_target = 2500

start_x = -112

start_distance = 2612 --also the best distance

function reward()
    local distance = x_target - data.x
    local reward
    if distance < start_distance then
        reward = 1/distance
        start_distance = distance
    else
        reward = -0.00001
    end
    return reward;
end