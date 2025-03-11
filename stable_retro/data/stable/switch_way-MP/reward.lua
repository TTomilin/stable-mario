y_target = 960
x_target = 224
start_x = 56
start_y = 1936
start_distance_tofin = math.sqrt((x_target-start_x)^2+(y_target-start_y)^2) --also the best distance_tofin
last_distance_tofin = start_distance_tofin
start_distance_tostart = 0 --also the best distance_tostart
last_distance_tostart = start_distance_tostart
gotten_reward = false

function reward()
    local distance_tofin = math.sqrt((x_target-data.x)^2+(y_target-data.y)^2)
    local distance_tostart = math.sqrt((start_x-data.x)^2+(start_y-data.y)^2)
    local reward = 0
    if distance_tofin < start_distance_tofin then
        reward = 0.004*(start_distance_tofin - distance_tofin) 
        start_distance_tofin = distance_tofin
    end
    reward = reward + 0.001*(last_distance_tofin - distance_tofin) -0.0001
    last_distance_tofin = distance_tofin
    if distance_tostart > start_distance_tostart then
        reward = 0.008*(-start_distance_tostart + distance_tostart) 
        start_distance_tostart = distance_tostart
    end
    reward = reward + 0.002*(-last_distance_tostart + distance_tostart) -0.0001
    last_distance_tostart = distance_tostart
    if gotten_reward == false and distance_tofin < 50 then
        reward = reward + 1 + 0.1 * data.time
        gotten_reward = true
    end
    reward = reward / 10
    return reward
end