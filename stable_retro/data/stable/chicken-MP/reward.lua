peachfin = false
mariofin = false
function reward()
    local reward = 0
    if data.xmario < 70 and not mariofin and peachfin then
        mariofin = true
        reward = 1
    elseif data.xmario < 70 and not mariofin and not peachfin then
        mariofin = true
        reward = -1
    elseif data.ingame == 0 then
        if mariofin and not peachfin then
            reward = 2
        elseif not mariofin then
            reward = -1
        end
    elseif data.xpeach > 169 then
            peachfin = true
    end
    return reward / 1.8
end