local macondo = require("macondo")

macondo.load('/tmp/livegame.gcg')
macondo.last()
local elite_play, details = macondo.elite_play()

if elite_play ~= nil then
    print("BEST:" .. elite_play)
    print("DETAILS:" .. details)
end