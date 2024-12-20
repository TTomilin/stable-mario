maxheight = 2880
time = 0
tottime = 0
lastx = 80
prevheight = 2880
function reward()
    local reward
    local x = data.xcoordinate
    local height = data.height
    if height < maxheight then
        time = 0
        reward = 0.0001*(maxheight-height)^2
        if (prevheight - height) > 8 then
            reward = reward + 0.001
        end
        maxheight = height
    else
        time = time + 1
        reward = -0.0000001*(time-50)
    end
    if x < lastx and height > 2327 and height < 2375 then
        reward = reward - 0.0001
    end
    lastx = x
    prevheight = height
    tottime = tottime + 1
    -- reward = reward + 0.00000001*(2880 - data.height)^2
    return reward
    
    -- if x < 80 then
    --     time = time + 1
    --     if time > 180 then
    --         if data.height < 2350 and x < 80 then
    --             reward = reward - 0.01
    --         else
    --             reward = reward - 0.00000003*(time - 180)*((x-40)^2)
    --         end
    --     end
    -- elseif x > 160 then
    --     time = time + 1
    --     if time > 180 then
    --         reward = reward - 0.00000003*(time - 180)*((x-200)^2)
    --     end
    -- else
    --     time = 0
    -- end
    
end