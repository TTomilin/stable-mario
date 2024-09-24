prevprogress = 0
function reward()
    local progress = data.score
    local reward = (progress - prevprogress)*0.0005 
    prevprogress = progress
    if reward == (200*0.0005) then
        reward = 0
    end
    return reward
end