previous_score = 0
time = 0

function reward()
    local score_rew = data.spooky_spike_score - previous_score
    previous_score = data.spooky_spike_score

    return score_rew;
end