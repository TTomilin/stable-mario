previous_score = 0
previous_health = 0
previous_speed = 0
cumulative_score = 0
go_counter = 0

function reward()
    -- compute score based on distance
    local score_incr = data.score - previous_score
    previous_score = data.score -- update previous score

    return score_incr;
end