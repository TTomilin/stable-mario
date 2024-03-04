previous_score = 0
time = 0

function reward()
    local score_rew = data.broom_zoom_score - previous_score
    previous_score = data.broom_zoom_score
    
    if score_rew > 0 then
        print(score_rew)
    end
    if data.game_over == 1 then
        print(string.format("episode reward: %d", previous_score))
    end

    return score_rew;
end