local macondo = require("macondo")

local fileutil = require("scripts.fileutil")

if args == nil or args[1] == nil then
    print("This script requires an argument - the CGP string")
    return
end

cgp = args[1]
print("Using provided cgp: ".. cgp)

macondo.load("cgp " .. cgp)
macondo.gen("40")
macondo.sim("-plies 5 -stop 99 -collect-heatmap true")
os.execute("sleep 1")

while macondo.busy() do
    os.execute("sleep 1")
end

local respshow = macondo.sim("show")
local respdetails = macondo.sim("details")


-- Preprocess respshow to extract the header and the rest of the content
local header, content = respshow:match("^(.-\n)(.+)$")
-- Adjust the pattern to correctly extract the winning play
local winning_play_line = content:match("^(.-)\n")

winning_play_line = winning_play_line and winning_play_line:match("^%s*(.-)$")
local winning_play = winning_play_line and winning_play_line:match("^(%S+ %S+)")
-- simulation results for top plays
local top_plays = header .. content:match("^(.-\n.-\n.-\n.-\n.-\n)")

-- Print or use the extracted values
-- print("Header:\n" .. (header or "Not found"))
-- print("Winning Play: " .. (winning_play or "Not found"))
-- print("Top Plays:\n" .. (top_plays or "Not found"))


-- Prepare a table to gather the desired output lines
local output_detail_lines = {}

-- These variables track which section we're in and count header and play lines
local currentSection = nil  -- Will be "Ply 1" or "Ply 2"
local headerCount = 0       -- Count header (title's own header lines after the section title)
local playCount = 0         -- Count of play data lines

-- Iterate over every line in the respdetails string
for line in respdetails:gmatch("([^\n]+)") do
  -- Check if the line marks the start of a new section
  if line:match("^%*%*Ply 1") then
    currentSection = "Ply 1"
    headerCount = 0
    playCount = 0
    table.insert(output_detail_lines, line)

  elseif line:match("^%*%*Ply 2") then
    currentSection = "Ply 2"
    headerCount = 0
    playCount = 0
    table.insert(output_detail_lines, "")  -- optional blank line before new section
    table.insert(output_detail_lines, line)

  elseif line:match("^%*%*Ply") then
    -- Found any other section header (like Ply 3, etc.) so if we're done with Ply 2 then break out.
    if currentSection == "Ply 2" then
      break  -- Stop reading further as we only want Ply 1 and Ply 2.
    end

  else
    -- Only process lines if we are inside Ply 1 or Ply 2 sections.
    if currentSection == "Ply 1" or currentSection == "Ply 2" then
      -- The first two lines after the section title are assumed to be the header and separator.
      if headerCount < 2 then
        table.insert(output_detail_lines, line)
        headerCount = headerCount + 1
      else
        -- Then output only the top 5 play lines.
        if playCount < 5 then
          table.insert(output_detail_lines, line)
          playCount = playCount + 1
        else
          -- For Ply 2, once we've got five play lines, we're done
          if currentSection == "Ply 2" then
            break
          end
        end
      end
    end
  end
end

local truncated_detail_lines = table.concat(output_detail_lines, "\n")

--

local playstats = macondo.sim("playstats \"" .. winning_play .. "\"")
local state = "NONE"      -- Current state: NONE, OPP, or OUR
local oppCount = 0        -- Count of play lines captured in "Opponent's next play"
local ourCount = 0        -- Count of play lines captured in "Our follow-up play"
local output_playstat_lines = {}

for line in playstats:gmatch("([^\n]+)") do
  -- Check for section headers no matter what the current state is
  if line:find("### Opponent's next play") then
    state = "OPP"       -- switching to Opponent's next play section
    oppCount = 0
    table.insert(output_playstat_lines, line)
  elseif line:find("### Our follow%-up play") then
    -- Note: if a dash appears in the actual text, escape it in the pattern
    state = "OUR"       -- switching to Our follow-up play section
    ourCount = 0
    table.insert(output_playstat_lines, "\n" .. line)  -- add a newline before new section header
  elseif state == "OPP" then
    -- When in Opponent's section:
    -- If this line is the table header (it begins with "Play"), capture it.
    if line:match("^%s*Play") then
      table.insert(output_playstat_lines, line)
    -- If it's a Bingo probability line, then end this section.
    elseif line:find("Bingo probability:") then
      state = "NONE"
    -- Otherwise, if we haven't yet captured 5 play lines, capture this line.
    elseif oppCount < 5 then
      table.insert(output_playstat_lines, line)
      oppCount = oppCount + 1
    end
  elseif state == "OUR" then
    -- For our follow-up section, similar handling
    if line:match("^%s*Play") then
      table.insert(output_playstat_lines, line)
    elseif line:find("Bingo probability:") then
      state = "NONE"
    elseif ourCount < 5 then
      table.insert(output_playstat_lines, line)
      ourCount = ourCount + 1
    end
  end
end

local truncated_winner_playstats = table.concat(output_playstat_lines, "\n")

-- print ("state:"..macondo.gamestate())

local situation = {
    game_state = macondo.gamestate(),
    sim_results = top_plays,
    sim_details = truncated_detail_lines,
    best_play = winning_play,
    winning_play_stats = truncated_winner_playstats,
}

situation_template = fileutil:read_file("explainer/situation_template.md")
prompt_template = fileutil:read_file("explainer/main_prompt.md")

situation_text = string.gsub(situation_template, "{([%w_]+)}", situation)

local prompt_vars = {
    situation = situation_text,
    best_play = winning_play
}

local prompt_text = string.gsub(prompt_template, "{([%w_]+)}", prompt_vars)
print("Prompt text:\n\n" .. prompt_text)