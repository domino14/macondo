-- This program analyses complete games.
-- You can find the most recent version of this program along with documentation here:
-- https://bitbucket.org/sitakr/macondostats/src/main/

local macondo = require("macondo")

local name = "xyz"
local genMoves = 25
local simSeconds = 3
local simPlies = 5
local performanceReference = 15

local letterRegex = "%aÄäÖöÜü"

local letterRegexWithBrackets = "[" .. letterRegex .. "]"

function sleep(seconds)
--	os.execute("ping -n " .. tonumber(seconds+1).. " localhost > NUL")
	os.execute("sleep " .. tonumber(seconds))
end

function findNameAndUnseenTileCount(board)
	local playerName = board:match("->%s+(%a+)%s+")
	local tileCount = board:match("Bag%s%+%sunseen%:%s%((%d+)%)")
	return playerName, tonumber(tileCount)
end

function setTurnAndGetBoard(turn)
	return macondo.turn(tostring(turn))
end

function round(number)
	return string.format("%.2f", number)
end

function playToSimResult(play)
	if play:find("exch") or play:find("Pass") then
		return play
	end
	local result = ""
	local inOverlap = false
	for i = 1, #play do
		local c = play:sub(i,i)
		local nextItem = c
		if inOverlap then
			if c ~= "." then 
				inOverlap = false
			else
				nextItem = ""
			end
		else
			if c == "." then 
				nextItem = "%([^%)]-%)"
				inOverlap = true
			end
		end
		result = result .. nextItem
	end
	return result
end

function calculateLossBySim(play, playInGen, tileCount)
	local equityMoves = macondo.gen(genMoves)
	local foundMove = equityMoves:match(string.format("(%s )", playInGen))
	if (foundMove == nil) then
		macondo.add(play)
	end
	if tileCount <= 14 then
		actualSimPlies = simPlies * 2
		if tileCount >= 7 then
			actualSimSeconds = simSeconds * 2
		else 
			actualSimSeconds = simSeconds
		end
	else
		actualSimSeconds = simSeconds
		actualSimPlies = simPlies
	end
	macondo.sim_async("-plies " .. actualSimPlies)
	sleep(actualSimSeconds)
	local simRes = macondo.sim_async("stop")
	while macondo.busy() do
		sleep(0.1)
	end
	local simResultPlay = playToSimResult(playInGen)
	local actualPlayRegex = "(" .. simResultPlay .. ")" .. "%s[%s%?" .. letterRegex .. "]+%d+%s*%s%s%s%s(%d*%.%d*)..%d*%.%d*%s*(%-?%d*%.%d*)"
	local actPlay, actWin, actEquity = simRes:match(actualPlayRegex)
	local equityRegex = "[%a%s%d]*%s*%s%s%s%s(%d*%.%d*)..%d*%.%d*%s*(%-?%d*%.%d*)"
	local bestWin, bestEquity = simRes:match(equityRegex)
	local winLoss = bestWin-actWin
	local equityLoss = bestEquity-actEquity
	local simLineRegex = "([%a%d%(]+%s[" .. letterRegex .. "%(%)]+)%s*[%?" .. letterRegex .. "]*%s%s%s*%d"
	local bestPlay = simRes:match(simLineRegex)
	local i = 0
	local refWin = nil
	local refEquity = nil
	local hasAscendingEquityByConstantWin = false
	for winPercentage, equity in simRes:gmatch(equityRegex) do
		i = i + 1
		if winPercentage == refWin and tonumber(equity) > tonumber(refEquity) then
			hasAscendingEquityByConstantWin = true
		end
		refWin = winPercentage
		refEquity = equity
		if i == performanceReference then
			break
		end
	end
	local referenceWinLoss = bestWin - refWin
	local referenceEquityLoss = bestEquity - refEquity
	local performance = 100
	if referenceWinLoss > 0 then
		performance = (1 - ((bestWin - actWin) / (referenceWinLoss))) * 100
	elseif hasAscendingEquityByConstantWin then
		performance = 100
	elseif referenceEquityLoss > 0 then
		performance = (1 - ((bestEquity - actEquity) / (referenceEquityLoss))) * 100
	end
	
	return winLoss, equityLoss, performance, actPlay, bestPlay
end


if args ~= nil and args[1] ~= nil then
	name=args[1]
end

local totalWinLoss = 0
local totalEquityLoss = 0
local totalPerformance = 0
local numberOfMoves = 0

for turn=0,90 do
	local board = setTurnAndGetBoard(turn)
	
	if board == nil or board:find("from their rack") then
		break
	end

	local playerName, tileCount = findNameAndUnseenTileCount(board)
	
	if playerName == name then
		nextBoard = setTurnAndGetBoard(turn+1)
		if nextBoard:find("from their rack") then
			break
		end
		if not nextBoard:find("challenged off") and (
				nextBoard:find(name .. " played ") or
				nextBoard:find(name .. " exchanged ") or 
				nextBoard:find(name .. " passed ")) then
			local play = nextBoard:match("played%s([%d%a]+%s[%." .. letterRegex .. "]+)%s")
			local possibleChallengBoard = setTurnAndGetBoard(turn+2)
			if (possibleChallengBoard ~= nil and possibleChallengBoard:find("challenged off")) or nextBoard:find("passed, holding") then
				play = "pass"
				playInGen = "%(Pass%)"
			else
				playInGen = play
				if play == nil then
					play = nextBoard:match("exchanged%s(" .. letterRegexWithBrackets .. "+)%sfrom")
					play = "exch " .. play
					playInGen = "%(" .. play .. "%)"
				else
					playInGen = play
				end
			end
			setTurnAndGetBoard(turn)
			winLoss, equityLoss, performance, actPlay, bestPlay = calculateLossBySim(play, playInGen, tileCount)
			print()
			print("--------------------------")
			numberOfMoves = numberOfMoves + 1
			if winLoss > 0 or equityLoss~=0 then
				print("turn " .. turn .. ", played '" .. actPlay .. "'")
				print("best play '" .. bestPlay .. "'")
				print("win loss: " .. round(winLoss))
				print("equity loss: " .. round(equityLoss))
				print("performance: " .. round(performance))
				
				totalWinLoss = totalWinLoss + winLoss
				totalEquityLoss = totalEquityLoss + equityLoss
			else
				print("turn " .. turn .. ", played '" .. actPlay .. "'" .. ", OK.")
			end
			totalPerformance = totalPerformance + performance
		end
	end
end

print()
print()
print("Total win percentages lost: " .. round(totalWinLoss))
print("Total equity lost: " .. round(totalEquityLoss))
print("Average performance: " .. round(totalPerformance/numberOfMoves))
