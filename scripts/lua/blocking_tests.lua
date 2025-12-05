local macondo = require("macondo")

local cgp = "4EXODE6/1DOFF1KERATIN1U/1OHO8YEN/1POOJA1B3MEWS/5SQUINTY2A/4RHINO1e3V/2B4C2R3E/GOAT1D1E2ZIN1d/1URACILS2E4/1PIG1S4T4/2L2R4T4/2L2A1GENII3/2A2T1L7/5E1A7/5D1M7 AEEIRUW/V 410/409 0 lex CSW24;"

local threads = 8

macondo.load('cgp ' .. cgp)

-- The endgame() function blocks until complete and returns the result directly.
local result = macondo.endgame(string.format('-plies 4 -threads %d -parallel-algo auto', threads))
print(result)

macondo.gen(20)
local sim_result = macondo.sim("-plies 5 -stop 99")
print(sim_result)

-- preendgame

macondo.load('cgp 15/3Q7U3/3U2TAURINE2/1CHANSONS2W3/2AI6JO3/DIRL1PO3IN3/E1D2EF3V4/F1I2p1TRAIK3/O1L2T4E4/ABy1PIT2BRIG2/ME1MOZELLE5/1GRADE1O1NOH3/WE3R1V7/AT5E7/G6D7 ENOSTXY/ACEISUY 356/378 0 lex NWL20;')

local result = macondo.peg()
print(result)

-- infer

macondo.new()
macondo.rack("PHEW")
macondo.commit("8F PHEW")
local result = macondo.infer("-time 10")
print ("infer result: " .. result .. "\nDone.")


