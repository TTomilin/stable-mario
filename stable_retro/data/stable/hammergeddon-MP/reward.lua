previous_lives_opp = 3
previous_lives_player = 3


function reward()
    local score_rew = previous_lives_opp - data.hp_opp -0.001 + data.hp_player - previous_lives_player
    previous_lives_opp = data.hp_opp
    previous_lives_player = data.hp_player
    return score_rew;
end