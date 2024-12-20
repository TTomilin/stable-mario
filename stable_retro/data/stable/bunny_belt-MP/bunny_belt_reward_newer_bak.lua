prev_action = 0
prev_score = 0
interrupted = false
count = 0

function reward()
    local reward = 0
    
    if prev_action == data.action then
        interrupted = true -- determine whether the model pressed an incorrect button
    end

    -- following if-statements captures all cases in which the previous action was NOT the last action for the current bunny
    if interrupted == false and prev_action < data.action and data.action > 20 then
        reward = reward + 10 -- reward 10 if model immediately pushes the right button after the last right button
        count = count + 1 -- increment number of correct actions
    elseif prev_action + 5 < data.action and data.action > 20 then -- we had +5 because sometimes 'action' jumps by 1 randomly
        count = count + 1 -- increment number of correct actions
    end

    -- following if statement captures all cases in which the previous action WAS the last action for the previous bunny
    if interrupted == false and prev_action > data.action and data.action <= 20 then
        reward = reward + 10 -- reward 10 if model immediately pushes the right button after the last right button
        count = count + 1 -- increment number of correct actions
    elseif prev_action > data.action and data.action <= 20 then
        reward = reward + 0 -- provide no reward if the model fails to press the correct button w.r.t. the last
        count = count + 1 -- increment number of correct actions
    end

    if prev_score < data.score then
        reward = reward + 100 -- reward 100 if the score of the model increases (only happens if you make one correct bunny)
    end

    if count == 5 then -- if bunny has been completed...
        interrupted = false -- set interrupted to false
        count = 0 -- set count to 0
    end

    prev_action = data.action
    prev_score = data.score

    return reward
end