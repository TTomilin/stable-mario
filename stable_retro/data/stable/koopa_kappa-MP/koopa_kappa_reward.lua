frame_counter = 0
current_bowser = 34
life = 0
function reward()
    frame_counter = frame_counter + 1
    local local_reward = 0
    local time_penalty = 0
    if data.num_bowser < current_bowser then 
        current_bowser = data.num_bowser
        local_reward = local_reward + 0.02
    end
    if data.life > life then 
        life = data.life
        local_reward = local_reward - 0.2
    end
    if frame_counter % 60 == 0 then
        time_penalty = -0.001
    end
    local_reward = local_reward + time_penalty
    return local_reward
end