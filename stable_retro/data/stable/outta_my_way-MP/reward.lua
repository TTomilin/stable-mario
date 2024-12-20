x_target = 980
y_target = 140
start_x = 48
start_y = 96
start_distance = math.sqrt((x_target-start_x)^2+(y_target-start_y)^2) --also the best distance

function reward()
    local distance = math.sqrt((x_target-data.x)^2+(y_target-data.y)^2)
    local reward
    if distance < start_distance then
        reward = 1/distance
        start_distance = distance
    else
        reward = 0
    end
    return reward;
end
