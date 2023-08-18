-- analyze a CGP "appropriately", depending on how many tiles are left.
-- CGP is parsed into args[1]

local res = macondo_load('cgp ' .. args[1])

print(res)
-- local res = macondo_gen(80)
