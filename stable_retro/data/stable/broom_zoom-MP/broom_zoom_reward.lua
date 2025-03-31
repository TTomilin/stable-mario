previous_score = 0
total = 0

function reward()
    local score_rew = 0

    score_rew = data.broom_zoom_score - previous_score
    previous_score = data.broom_zoom_score

    return score_rew;
end