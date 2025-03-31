
frame_counter = 0
last_mario_health = 3

function reward()
    frame_counter = frame_counter + 1
    local local_reward = 0
    local time_penalty = 0
    local health_penalty = 0

    if frame_counter % 60 == 0 then
        time_penalty = -0.001
    end

    if data.Mario < last_mario_health then
        health_penalty = -0.5
        last_mario_health = data.Mario
    end

    local_reward = local_reward + time_penalty + health_penalty
    return local_reward
end
