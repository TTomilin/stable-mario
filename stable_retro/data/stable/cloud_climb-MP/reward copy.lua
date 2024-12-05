function reward()
    local height = data.height
    local reward
    local maxheight
    if maxheight ~= nil then
        height = maxheight
    end
    if height < maxheight then
        maxheight = height
        reward = 0.01
    elseif height - 100 < maxheight then
        reward = -0.0001
    else
        reward = -0.0001 +(maxheight - height + 100)/10000
    end
end