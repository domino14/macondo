local fileutil = {}
-- Function to read an entire file
function fileutil:read_file(filepath)
    local file, err = io.open(filepath, "r")
    if not file then
        error("Could not open file: " .. filepath .. " (" .. err .. ")")
    end
    local content = file:read("*all")
    file:close()
    return content
end

return fileutil
