previous_stage = 1
stage_mod = 10
previous_count = 0
count_mod = 100000000

function reward()
    local reward = 0
    
    -- check if progress was made:
    local current_count = data.count / count_mod
    if current_count > previous_count then
        reward = reward + 1 -- reward progress
        previous_count = current_count
    end

    -- check if model progressed to a next state:
    local current_stage = math.fmod(data.stage, stage_mod)
    if current_stage > previous_stage then
        previous_count = 0 -- reset progress
        previous_stage = current_stage
    end

    return reward
end