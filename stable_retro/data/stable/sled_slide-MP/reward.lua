lastsegment = 0
notpassed13 = true
highestsegment = 0
previousspeed = 0
gotreward = false
leeway = 0
function reward()
    local reward = 0
    local progress = data.progress
    if progress == 13 then
        notpassed13 = false
    end
    if leeway ~= 0 then
        leeway = leeway - 1
    end
    if progress == 14 and notpassed13 then
        reward = -0.0001
    elseif gotreward and leeway == 0 then
        reward = -0.01
    elseif (progress == 0) and (not notpassed13) and (not gotreward) then
        reward = 17
        gotreward = true
        leeway = 4
    elseif progress < highestsegment then
        reward = -0.0001
    else
        reward = 1*(progress - lastsegment) + 0.0000001*(data.speed) + 0.005*(data.speed - previousspeed)
        previousspeed = data.speed
        if progress > lastsegment then 
            highestsegment = progress
        end
        lastsegment = progress
    end 
    reward = reward / 17
    return reward 
end

