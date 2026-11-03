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

const caliper = run.resultSummary([
  { name: "Diameter", text: "83.92768" },
  { name: "Width", text: "140" },
  { name: "Height", text: "90" },
  { name: "Radius", text: "41.96384" },
]);
assert(caliper === "直徑 83.93 · 寬 140 · 高 90", "a caliper run names diameter, width, and height");
const judged = run.resultSummary([
  { name: "Judge", text: "'不合格'" },
  { name: "Width", text: "22" },
]);
assert(judged === "判定 不合格 · 寬 22", "a failed tolerance is named before the width");
assert(run.resultSummary([{ name: "Width", text: "22" }]) === "寬 22", "a stripe run names the width");
const defect = run.resultSummary([
  { name: "Count", text: "1" },
  { name: "BrightCount", text: "1" },
  { name: "Area", text: "[3550]" },
  { name: "BrightArea", text: "[144]" },
  { name: "Row", text: "[29.5]" },
]);
assert(defect === "個數 1 · 亮點 1 · 面積 3550 · 亮點面積 144", "a defect run names both counts and areas");
assert(run.resultSummary([{ name: "Area", text: "[2988, 100, 20]" }]) === "", "a long area list stays out of the status line");
const code = run.resultSummary([
  { name: "Text", text: "'4006381333931'" },
  { name: "QrText", text: "'VISDEV-3'" },
  { name: "Count", text: "1" },
]);
assert(code === "條碼 4006381333931 · 二維碼 VISDEV-3 · 個數 1", "barcode text drops the quotes");
const locate = run.resultSummary([
  { name: "Count", text: "4" },
  { name: "Score", text: "[0.97, 0.91, 0.95, 0.93]" },
]);
assert(locate === "個數 4 · 最低分 0.91", "shape scores report the lowest one");
assert(run.resultSummary([]) === "", "a program without those names adds nothing");
const line = run.resultSummary([
  { name: "Judge", text: "'不合格'" },
  { name: "OkCount", text: "3" },
  { name: "NgCount", text: "2" },
  { name: "Total", text: "5" },
  { name: "CsvFile", text: "'workspace/line_result.csv'" },
  { name: "Count", text: "1" },
]);
assert(line === "判定 不合格 · 合格 3 · 不合格 2 · 總數 5 · 明細 workspace/line_result.csv", "a line run reports pass and fail counts");
const running = run.liveLine("專案 demo · ", 18, [
  { name: "Judge", text: "'不合格'" },
  { name: "OkCount", text: "1" },
  { name: "NgCount", text: "1" },
  { name: "Total", text: "2" },
  { name: "Count", text: "0" },
]);
assert(running === "專案 demo · 下一行 18 · 判定 不合格 · 合格 1 · 不合格 1 · 總數 2", "a running line shows the counts");
assert(run.liveLine("", null, []) === "執行中", "running without a line still says it is running");
const lastNg = run.resultSummary([
  { name: "Judge", text: "'不合格'" },
  { name: "OkCount", text: "3" },
  { name: "NgCount", text: "2" },
  { name: "Total", text: "5" },
  { name: "LastNg", text: "'ng_02.png'" },
  { name: "CsvFile", text: "'workspace/line_result.csv'" },
]);
assert(lastNg === "判定 不合格 · 合格 3 · 不合格 2 · 總數 5 · 不合格圖 ng_02.png", "a line run names the latest failed image");
assert(run.namedText([
  { name: "LastNg", text: "'ng_02.png'" },
  { name: "CsvFile", text: "'workspace/line_result.csv'" },
], "LastNg") === "ng_02.png", "the failed image name drops the quotes");
assert(run.namedText([], "CsvFile") === "", "a missing file name stays blank");
const color = run.resultSummary([
  { name: "Judge", text: "'合格'" },
  { name: "Count", text: "1" },
  { name: "Hue", text: "60" },
  { name: "Area", text: "[1200]" },
]);
assert(color === "判定 合格 · 個數 1 · 色調 60 · 面積 1200", "a color run names hue and area");
const chars = run.resultSummary([
  { name: "Judge", text: "'合格'" },
  { name: "Chars", text: "'VISDEV 168'" },
  { name: "Expect", text: "'VISDEV 168'" },
  { name: "Count", text: "1" },
  { name: "Score", text: "[0.91]" },
]);
assert(chars === "判定 合格 · 文字 VISDEV 168 · 預期 VISDEV 168 · 個數 1 · 最低分 0.91", "a character run names the text and the expected word");
assert(run.resultSummary([{ name: "Register", text: "2" }]) === "暫存器 2", "a modbus read names the register");

console.log("run source ok");
