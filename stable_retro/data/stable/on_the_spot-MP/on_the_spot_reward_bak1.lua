previous_stage = 1

function reward()
    local reward = 0
    
    -- check if a mistake was made:
    if data.error_3 == 90402614005800 then
        reward = reward - 1 -- punish model for a misstep
    end

    -- check if model progressed to a next state:
    local current_stage = math.fmod(data.stage, 10)
    if current_stage > previous_stage then
        previous_stage = current_stage
        reward = reward + 5
    end

    return reward
end