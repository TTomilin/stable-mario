turn = 0
round = 1
previous_turn_mario = 0
previous_turn_peach = 0

function reward()
    local reward = 0
    local shells_mario = data.shell1 + data.shell3 + data.shell5
    local shells_peach = data.shell2 + data.shell4 + data.shell6
    if data.turn == 0 then
        previous_turn_mario = 0
        previous_turn_peach = 0
    elseif shells_mario ~= previous_turn_mario or shells_peach ~= previous_turn_peach then
        if shells_mario % 5 == 3 then
            shells_mario = (shells_mario + 2) * 4
        elseif shells_mario % 5 == 4 then
            shells_mario = (shells_mario + 1) * 2
        end
        if shells_peach % 5 == 3 then
            shells_peach = (shells_peach + 2) * 4
        elseif shells_peach % 5 == 4 then
            shells_peach = (shells_peach + 1) * 2
        end
        reward = (shells_mario - previous_turn_mario - shells_peach + previous_turn_peach) / 100
        previous_turn_mario = data.shell1 + data.shell3 + data.shell5
        previous_turn_peach = data.shell2 + data.shell4 + data.shell6
    end
    return reward
end

