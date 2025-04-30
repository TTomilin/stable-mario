yfin = -1150
ystart = -112
bestdistance = -yfin+ystart
lastdistance = bestdistance
time = 60
nothadreward = true
function reward()
    local reward = 0
    if nothadreward then
        local distance = - yfin + data.y
        if distance < bestdistance then
            reward = 0.0008*(bestdistance - distance)
            bestdistance = distance
        end
        if data.y < -1150 then
            reward = (reward + 1)
            nothadreward = false
        end
        reward = reward + 0.0002*(lastdistance - distance) + 0.01*(data.time-time)
        time = data.time
        lastdistance = distance
    else 
        reward = -0.001
    end
    return reward / 1.76
end