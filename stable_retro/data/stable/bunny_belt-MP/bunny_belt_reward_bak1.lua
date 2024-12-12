prev_action = 0
prev_score = 0
interrupted = false

function reward()
    local reward = 0
    
    if prev_action == data.action then
        interrupted = true -- determine whether the model pressed an incorrect button
    end

    -- following if-statements captures all cases in which the previous action was NOT the last action for the current bunny
    if interrupted == false and prev_action < data.action and data.action > 20 then
        reward = reward + 10 -- reward 10 if model immediately pushes the right button after the last right button
    elseif prev_action + 5 < data.action and data.action > 20 then -- we had +5 because sometimes 'action' jumps by 1 randomly
        reward = reward + 1 -- reward 1 if the model presses the right button, but not immediately after the previous right button
        interrupted = false -- reset interrupted
    end

    -- following if statement captures all cases in which the previous action WAS the last action for the previous bunny
    if interrupted == false and prev_action > data.action and data.action <= 20 then
        reward = reward + 10 -- reward 10 if model immediately pushes the right button after the last right button
    elseif prev_action > data.action and data.action <= 20 then
        reward = reward + 1 -- reward 1 if the model presses the right button, but not immediately after the previous right button
        interrupted = false -- resit interrupted
    end

    if prev_score < data.score then
        reward = reward + 100 -- reward 100 if the score of the model increases (only happens if you make one correct bunny)
    end

    prev_action = data.action
    prev_score = data.score

    return reward
end