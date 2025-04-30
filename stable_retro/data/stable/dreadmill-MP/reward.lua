xfin = 2500
xstart = -112
bestdistance = xfin-xstart
lastdistance = bestdistance
nothadreward = true
function reward()
    local reward = 0
    if nothadreward then
        local distance 
        distance = xfin - data.x
        if distance < bestdistance then
            reward = 0.0008*(bestdistance - distance)
            bestdistance = distance
        end
        if data.x > 2499 then
            reward = reward + 1
            nothadreward = false
        end
        reward = reward + 0.0002*(lastdistance - distance) - 0.0001
        lastdistance = distance
    else 
        reward = -0.001
    end
    return reward / 3.4
end