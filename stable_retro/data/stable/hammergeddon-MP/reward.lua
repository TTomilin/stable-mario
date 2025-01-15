previous_lives_opp = 3
previous_lives_player = 3


function reward()
    local reward = 0
    if previous_lives_opp > data.hp_opp then
        reward = 1
        previous_lives_opp = previous_lives_opp - 1
    end
    if previous_lives_player > data.hp_player then
        reward = -1
        previous_lives_player = previous_lives_player - 1
    end
    return reward
end