previous_score = 0
time = 0

function reward()
    local score_rew = data.broom_zoom_score - previous_score
    previous_score = data.broom_zoom_score

    if data.game_over == 1 then
        print(string.format("episode reward: %d", previous_score)) -- just for debugging: wanted to see if average episode score matches the one displayed by stable-baselines
    end

    return score_rew;
end