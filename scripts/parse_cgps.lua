-- this script was used for a Kaggle competition
-- https://www.kaggle.com/c/scrabble-point-value/leaderboard
-- we got 4th place!

local function split(str, sep)
    local result = {}
    local regex = ("([^%s]+)"):format(sep)
    for each in str:gmatch(regex) do
       table.insert(result, each)
    end
    return result
end

string.startswith = function(self, str)
    return self:find('^' .. str) ~= nil
end


outfile = io.open("./scripts/out.txt", "w")
outfile:write("game_id,points\n")

error_score = 27       -- input data file had some errors

-- cgp files were created from the input data (that script isn't available
-- here but can be replicated with some work)

local function opcodeval(cgp, opcode)
    local line = split(cgp, " ")
    -- opcodes start at index 5, odd indices
    for i=5,9,2 do
        if line[i] == opcode then
            local unstripped = line[i+1]
            return unstripped:sub(1, -2)
        end
    end
end

local skillz = {
    [2000]=2,
    [1900]=3,
    [1800]=4,
    [1700]=5,
    [1600]=6,
    [1500]=7,
    [1400]=9,
    [1300]=20,
}


for line in io.lines('/Users/cesar/Downloads/cgps (2).txt') do
    local err = macondo_load('cgp ' .. line)
    if string.startswith(err, "ERROR: ") then
        local gid = opcodeval(line, "gid")
        outfile:write(gid .. "," .. error_score .. "\n")
    else
        -- note: rating isn't an opcode in CGP but we hacked this together.
        local movepos = skillz[tonumber(opcodeval(line, "rating"))]
        local gid = macondo_gid()
        local res = macondo_gen(tostring(math.floor(movepos)))
        -- ignore header row
        local lines = split(res, "\n")
        local the_move = lines[movepos + 1] -- lua is 1-indexed, so this is the movepos move
        -- write the game id
        outfile:write(gid .. ",")
        -- write the 1st element from the end of the table; this is the score.
        local fields = split(the_move, " ")
        outfile:write(fields[#fields-1] .. "\n")
    end
end

outfile:close()