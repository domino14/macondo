local macondo = require("macondo")

if args == nil or args[1] == nil then
    print("Usage: nps_bench.lua <CGP string>")
    return
end

local cgp = args[1]
local runs = 10

print("Benchmarking NPS over " .. runs .. " runs")
print("CGP: " .. cgp)

local total_nps = 0
local total_nodes = 0
local total_elapsed = 0

for i = 1, runs do
    macondo.load("cgp " .. cgp)
    macondo.gen(40)

    local t0 = os.clock()
    macondo.sim("-plies 5 -stop 99")
    local elapsed = os.clock() - t0

    local nodes = macondo.sim_nodes()
    local nps = nodes / elapsed

    print(string.format("run %2d: elapsed=%.3fs  nodes=%d  nps=%.0f",
        i, elapsed, nodes, nps))

    total_nps = total_nps + nps
    total_nodes = total_nodes + nodes
    total_elapsed = total_elapsed + elapsed
end

print(string.format("\naverage: elapsed=%.3fs  nodes=%.0f  nps=%.0f",
    total_elapsed / runs,
    total_nodes / runs,
    total_nps / runs))
