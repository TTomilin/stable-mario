current_score = 50
frame_counter = 0
function reward()
    frame_counter = frame_counter + 1
    local local_reward = 0
    local time_penalty = 0
    if data.score <current_score then 
        current_score = data.score
        local_reward = local_reward + 0.02
    end
    if frame_counter % 60 == 0 then
        time_penalty = -0.001
    end
    local_reward = local_reward + time_penalty
    return local_reward
end