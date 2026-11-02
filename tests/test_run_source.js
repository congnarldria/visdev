const run = require("../app/static/run_source.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

const source = "Vizion.read_image (Image, 'a.png')\nVizion.threshold (Image, Region, 128, 255)\n";
assert(run.sameProgram(source, source), "the editor matches the running program");
assert(!run.sameProgram(source, source + "A := 1\n"), "a new line is a different program");
assert(run.arrowLine(source, source, 2) === 2, "the arrow stays when the text matches");
assert(run.arrowLine(source, "Vizion.read_image (Image, 'a.png')\n* 註解\n", 1) === 1, "an untouched line still marks that row");
assert(run.arrowLine(source, "* 插入\n" + source, 1) === null, "a shifted line does not mark a different statement");
assert(run.arrowLine(null, "changed", 4) === 4, "without a run, the editor line is used");
assert(run.statusText(source, source, 2, true) === "", "a matching editor needs no extra note");
const note = run.statusText(source, source + "A := 1\n", 2, true);
assert(note.indexOf("第 2 行") === 0 && note.indexOf("按下執行時的程式") > 0, "a running edit names the line and the snapshot");
assert(run.statusText(source, "A := 1\n", 2, false).indexOf("再按執行或單步") > 0, "after the run, the next start uses the editor");
const paused = run.statusText(source, "A := 1\n", 3, false, "breakpoint");
assert(paused.indexOf("第 3 行") === 0 && paused.indexOf("中斷點") > 0, "a breakpoint stays on the program that was started");

console.log("run source ok");
