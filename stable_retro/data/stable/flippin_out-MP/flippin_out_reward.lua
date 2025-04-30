-- note: we use Manhattan distance, Euclidean distance would be inefficient

closest_distance = 728 - 56-- initialize at start distance

target_x = 728 -- set to destination x
target_y = 200 -- set to destination y
x_step = 48 -- step size in x direction
y_step = 48 -- step size in y direction


function reward()
    local score = 0 -- init score
    
    local current_distance = ((math.abs(data.x - target_x)) / x_step) + ((math.abs(data.y - target_y)) / y_step) -- compute manhattan distance to target
    if current_distance < closest_distance then -- check if new best has been achieved
        score = (closest_distance - current_distance) - 0.01 -- reward model for getting closer
        score = score / 1328.33
        closest_distance = current_distance -- update closest_distance
    end
    
    return score / 7
end