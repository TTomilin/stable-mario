tottime = 0
finished = 0
maxheight = 2880
timenorecord = 0
function reward()
    local reward = 0
    if finished == 0 then
        timenorecord = timenorecord + 1
        local height = data.height
        if height < maxheight then
            reward = 0.001*(maxheight - height)
            maxheight = height
            timenorecord = 0
        end
        -- reward = -0.0000000000001*(height - 2700 - ((1897-2880)/2800)*tottime)^3
        -- if timenorecord > 899 and maxheight == 2327 and data.xcoordinate < 100 and data.height < 2400 then
        --     reward = -0.2
        -- end
        if height < 2200 and reward > 0 then
            reward = reward*4
        elseif height < 2200 and reward < 0 then
            reward = reward*0.2
        end
        if height < 1900 and data.xcoordinate < 140 and data.xcoordinate > 100 then
            reward = reward + (3600-tottime)/1000
            finished = 1
        end
        tottime = tottime + 1
    else
        reward = -0.000001
    end  
    return reward
end

function endcondition()
    -- return data.ingame == 0 or (timenorecord > 900 and maxheight == 2327 and data.xcoordinate < 100 and data.height < 2400)
    return data.ingame == 0
end