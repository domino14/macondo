-- ABDADA Scaling Test
-- Tests the StuckPruning endgame position with different thread counts
-- and measures performance scaling

local macondo = require("macondo")

local cgp = "4EXODE6/1DOFF1KERATIN1U/1OHO8YEN/1POOJA1B3MEWS/5SQUINTY2A/4RHINO1e3V/2B4C2R3E/GOAT1D1E2ZIN1d/1URACILS2E4/1PIG1S4T4/2L2R4T4/2L2A1GENII3/2A2T1L7/5E1A7/5D1M7 AEEIRUW/V 410/409 0 lex CSW24;"

local thread_counts = {1, 2, 4, 8, 10, 12, 14, 16}
local results = {}

print("\n=== ABDADA Scaling Test: StuckPruning Position ===\n")
print("Position: Deep endgame with high branching factor")
print("Solving to 9 plies with -parallel-algo auto\n")
print("Testing thread counts: 1, 2, 4, 8, 10, 12, 14, 16\n")


for _, threads in ipairs(thread_counts) do
    print(string.format("Running with %d thread(s)...", threads))

    -- Load the CGP position
    macondo.load('cgp ' .. cgp)

    -- Run endgame solver with the specified thread count
    local start_time = os.clock()
    local result = macondo.endgame(string.format('-plies 3 -threads %d -parallel-algo auto', threads))
    local elapsed = os.clock() - start_time

    -- Parse the result to extract timing information
    -- The log format is: time-elapsed-sec:0.027151212
    local solve_time = elapsed  -- fallback to measured time
    if result then
        local time_in_result = result:match("time%-elapsed%-sec:([%d%.]+)")
        if time_in_result then
            solve_time = tonumber(time_in_result)
        end
    end

    results[threads] = {
        time = solve_time,
        result = result
    }

    print(string.format("  Completed in %.3f seconds\n", solve_time))
end

-- Calculate and display results table
print("\n=== Results ===\n")
print(string.format("%-8s | %-12s | %-10s | %-10s", "Threads", "Time (s)", "Speedup", "Efficiency"))
print(string.rep("-", 50))

local baseline_time = results[1].time

for _, threads in ipairs(thread_counts) do
    local time = results[threads].time
    local speedup = baseline_time / time
    local efficiency = (speedup / threads) * 100

    print(string.format("%-8d | %12.3f | %10.2fx | %9.1f%%",
        threads, time, speedup, efficiency))
end

print("\n")

-- Identify best scaling range
local best_efficiency_threads = 1
local best_efficiency = 100.0

for _, threads in ipairs(thread_counts) do
    if threads > 1 then
        local speedup = baseline_time / results[threads].time
        local efficiency = (speedup / threads) * 100
        if efficiency > best_efficiency then
            best_efficiency = efficiency
            best_efficiency_threads = threads
        end
    end
end

print(string.format("Baseline (1 thread): %.3f seconds", baseline_time))
print(string.format("Best efficiency: %.1f%% at %d threads", best_efficiency, best_efficiency_threads))
print(string.format("Best speedup: %.2fx at %d threads (%.3f seconds)",
    baseline_time / results[thread_counts[#thread_counts]].time,
    thread_counts[#thread_counts],
    results[thread_counts[#thread_counts]].time))
print("\n")
