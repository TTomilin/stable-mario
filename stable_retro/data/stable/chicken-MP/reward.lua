finished = false
total = 0
function reward()
    local reward
    if finished then
        reward = 0
    elseif data.winn == 1 then
        reward = 1
        total = total + 1
        finished = true
    end
    reward = -0.001
    return reward
end