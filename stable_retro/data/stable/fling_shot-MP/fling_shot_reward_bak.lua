prev_x_coord = -2621440
prev_score = -127

function reward()
    local x_coord = data.x_coord
    local score = data.score

    local reward = 0

    -- reward build up
    if score == -127 and x_coord < prev_x_coord then
        reward = reward + (prev_x_coord - x_coord) * 0.0015 -- caps it at around 2000 score
    end

    -- reward distance travelled
    if score ~= -127 and prev_score < score then
        reward = reward + (score - prev_score)
    end

    -- update globals
    prev_x_coord = x_coord
    prev_score = score

    return reward
end