local macondo = require("macondo")

cgp = '3ARABIC6/8UKE1T2/6REpINNED1/3JOG1X4R2/4WIMP4R2/2FIEF1I4A2/7R4N2/4N2E4E2/'..
    '3VATUS4S2/4T10/4T10/3OY10/3W11/OY1N11/DAGS11 ABEEIOS/ 211/242 0 lex NWL23;'

macondo.load("cgp " .. cgp)
macondo.gen("10")
macondo.add("C10 ISBA")
macondo.sim("-autostopcheckinterval 64 -fixedsimiters 100 -fixedsimplies 2 "..
            "-fixedsimcount 1000 -stop 99")

