// Assume add.wasm file exists that contains a single function adding 2 provided arguments
const fs = require('fs');

require('./wasm_exec');

const macondoPromise = new Promise((res, rej) => {
    const w = globalThis;
    w.resMacondo = res;
    w.rejMacondo = rej;
});

const wasmBuffer = fs.readFileSync('/home/cesar/code/liwords/liwords-ui/public/wasm/macondo.wasm');
const go = new globalThis.Go();

async function thingy() {
    let wasmModule;
    try {   
        wasmModule = await WebAssembly.instantiate(wasmBuffer, go.importObject)
    } catch (e) {
        console.log('couldnt load', e);
    }
  // Exported function live under instance.exports
  console.log('here1')
  try {
    console.log('here1.5')
    go.run(wasmModule.instance);
    console.log('here2')
  } catch (e) {
    console.log('couldnt run', e);
  }

  const lol = await macondoPromise;
  console.log('lol', lol)
 const CSW21 = fs.readFileSync('/home/cesar/code/liwords/liwords-ui/public/wasm/CSW19.kwg');
 const CSWLeaves = fs.readFileSync('/home/cesar/code/liwords/liwords-ui/public/wasm/CSW21.klv2');
 const english = fs.readFileSync('/home/cesar/code/macondo/data/letterdistributions/english');
 const winpct = fs.readFileSync('/home/cesar/code/macondo/data/strategy/default_english/winpct.csv');
 console.log(lol.precache("data/lexica/gaddag/CSW21.kwg", CSW21))
 console.log(lol.precache("data/strategy/CSW21/leaves.klv2", CSWLeaves))
 console.log(lol.precache("data/letterdistributions/english", english))
 lol.precache("data/strategy/default_english/winpct.csv", winpct);
// winpctfile:CSW:winpct.csv:
//  pegfile:NWL20:preendgame.json:
  const analyzer = await lol.newAnalyzer();
  console.log('analyzer is', analyzer);

 const analysis = await lol.analyzerAnalyze(analyzer, `{
    "scores": [0, 0],
    "onturn": 0,
    "size": 15,
    "rack": "EINRSTZ",
    "lexicon": "CSW21",
    "board": [
      "...............",
      "...............",
      "...............",
      "...............",
      "...............",
      "...............",
      "...............",
      "...HELLO.......",
      "...............",
      "...............",
      "...............",
      "...............",
      "...............",
      "...............",
      "..............."
    ]}`);

  console.log('analysis', analysis);

  lol.simInit(analyzer);
  
  let start = +new Date();
  lol.simSingleThread(analyzer, 1000);
  let end = +new Date();
  console.log(lol.simState(analyzer));
  console.log('time', end-start, 'milliseconds')
}

thingy();

//   instance.then(i => {
//     const { precache, simSingleThread } = i.exports;
//     console.log('precache is', precache)
//     const NWL20 = fs.readFileSync('/home/cesar/code/liwords/liwords-ui/public/wasm/NWL20.kwg');
//     const NWLleaves = fs.readFileSync('/home/cesar/code/liwords/liwords-ui/public/wasm/english.klv2');

//     const ret = precache("kwg:NWL20", NWL20);
//     const ret2 = precache("leavefile:NWL20:leaves.klv2", NWLleaves);
//         console.log(ret);
//         console.log(ret2);
//   });
