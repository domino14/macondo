function sleep(s)
  local ntime = os.clock() + s/10
  repeat until os.clock() > ntime
end

-- sim a position very quickly (just a tiny number of iterations)
macondo_set('lexicon NWL20')
macondo_load('xt 39444')
macondo_turn('14')
macondo_gen('5')
-- turn on logging, and sim on one core, then quit.
macondo_sim('log')  
macondo_sim('2 1') 
sleep(1) -- 1/10th of a sec
macondo_sim('stop') -- stop right away
