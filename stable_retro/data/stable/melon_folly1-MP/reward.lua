first = true
prev_progress = 0
function reward()
    local progress = data.score
    if first then
        first = false
        prev_progress = progress
    end
    local reward = (progress - prev_progress)*0.0005 
    prev_progress = progress
    if reward == 200*0.0005 then
        reward = 0
    end
    return reward
end
