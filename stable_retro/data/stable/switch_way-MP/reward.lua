x_target = 960
y_target = 224
start_x = 56
start_y = 1936
start_distance = math.sqrt((x_target-start_x)^2+(y_target-start_y)^2) --also the best distance
last_distance = start_distance

function reward()
    local distance = math.sqrt((x_target-data.x)^2+(y_target-data.y)^2)
    local reward = 0
    if distance < start_distance then
        reward = 0.004*(start_distance - distance) 
        start_distance = distance
    end
    reward = reward + 0.001*(last_distance - distance) -0.0001
    last_distance = distance
    return reward
end