score = 30
hold = false
function reward()
    local reward = -0.0001
    local change = score - data.score
    if data.blue + data.yellow + data.green + data.orange + data.pink == change and not hold then
        hold = true
    elseif not hold then
        reward = reward + change
        score = data.score
    elseif change ~= 0 then
        reward = reward + change
        score = data.score
        hold = false
    end
    return reward / 29
end

