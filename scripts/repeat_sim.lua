local macondo = require("macondo")


-- does not work on Windows, I think:
function sleep(n)
    os.execute("sleep " .. tonumber(n))
end

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

for i=1,500 do
    macondo.load('cgp 14Q/13GI/9A1B1U1/9C1O1C1/9TUPIK1/9I5/4R4V3OE/3JIBED1E3uH/2LID1WOOSH2T1/3V1A4I2G1/3EFT1cANNULAE/3YAR4T2Z1/4XI3PENNED/5A3ER2DO/7TANSY2L AEINOST/ 378/316 0 lex NWL23;')
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
