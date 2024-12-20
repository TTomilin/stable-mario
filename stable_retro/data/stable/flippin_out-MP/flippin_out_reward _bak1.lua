-- note: we use Manhattan distance, Euclidean distance would be inefficient

previous_x = 3670016 -- initialize at start x
previous_y = 4284219392 -- initialize at start y 
previous_distance = math.abs(3670016 - 48103424) / (6291456) + math.abs(4284219392 - 4284612863) / (3145728) -- initialize at start distance
closest_distance = math.abs(3670016 - 48103424) / (6291456) + math.abs(4284219392 - 4284612863) / (3145728) -- initialize at start distance

target_x = 48103424 -- set to destination x
target_y = 4284612863 -- set to destination y
x_step = 6291456 -- step size in x direction
y_step = 3145728 -- step size in y direction


function reward()
    -- compute score based on distance
    local current_distance = math.abs(data.x_pos - target_x) / x_step + math.abs(data.y_pos - target_y) / y_step -- compute manhattan distance to target
    local distance_change = current_distance - previous_distance -- find distance
    local score = -1 * distance_change -- reward decreases in distance

    -- update previous
    previous_distance = current_distance

    return score;
end