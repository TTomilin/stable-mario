blue = true
yellow = true
green = true
red = true
function reward()
    local reward = -0.0001
    if blue and data.blue == 1 then
        blue = false
        reward = reward + 1
    elseif yellow and data.yellow == 1 then
        yellow = false
        reward = reward + 2
    elseif green and data.green == 1 then
        green = false
        reward = reward + 3
    elseif red and data.red == 1 then
        red = false
        reward = reward + 1
    elseif not blue and data.blue == 0 then
        blue = true
        reward = reward - 1.1
    elseif not yellow and data.yellow == 0 then
        yellow = true
        reward = reward - 2.1
    elseif not green and data.green == 0 then
        green = true
        reward = reward - 3.1
    elseif not red and data.red == 0 then
        red = true
        reward = reward - 1.1
    end
    return  * 10
end