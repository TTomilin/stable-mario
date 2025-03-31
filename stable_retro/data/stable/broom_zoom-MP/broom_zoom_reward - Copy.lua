previous_score = 0
total = 0

function reward()
    local score_rew = 0

    if data.broom_zoom_score > previous_score then
        score_rew = 1
        total = total + score_rew
    end
    -- if data.game_over == 1 then
    --     print(string.format("episode reward: %d", total))
    -- end
    previous_score = data.broom_zoom_score

    return score_rew;
end