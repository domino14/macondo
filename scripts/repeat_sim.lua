local macondo = require("macondo")

if args == nil or args[1] == nil then
    print("This script requires an argument - the CGP string")
    return
end

cgp = args[1]
print("Using provided cgp: ".. cgp)

function dump(o)
    if type(o) == 'table' then
       local s = '{ '
       for k,v in pairs(o) do
          if type(k) ~= 'number' then k = '"'..k..'"' end
          s = s .. '['..k..'] = ' .. dump(v) .. ','
       end
       return s .. '} '
    else
       return tostring(o)
    end
 end


local plays = {}
start_time = os.time()

for i=1,100 do
    macondo.load("cgp " .. cgp)
    local elite_play, details = macondo.elite_play()
    if plays[elite_play] == nil then
        plays[elite_play] = 1
    else
        plays[elite_play] = plays[elite_play] + 1
    end
    print("WINNER ".. elite_play)
end

end_time = os.time()
elapsed_time = os.difftime(end_time, start_time)

print(dump(plays))
print("elapsed_time = " .. elapsed_time)
