-- note: we use Manhattan distance, Euclidean distance would be inefficient

closest_distance = math.abs(3670016 - 48103424) / (6291456) + math.abs(4284219392 - 4284612863) / (3145728) -- initialize at start distance

target_x = 48103424 -- set to destination x
target_y = 4284612863 -- set to destination y
x_step = 6291456 -- step size in x direction
y_step = 3145728 -- step size in y direction


function reward()
    local score = 0 -- init score
    
    local current_distance = ((math.abs(data.x_pos - target_x)) / x_step) + ((math.abs(data.y_pos - target_y)) / y_step) -- compute manhattan distance to target
    if current_distance < closest_distance then -- check if new best has been achieved
        score = (closest_distance - current_distance) -- reward model for getting closer
        closest_distance = current_distance -- update closest_distance
    end
    
    return score;
end