const tol = require("../app/static/tolerance.js");
const fs = require("fs");
const path = require("path");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

const caliper = fs.readFileSync(path.join(__dirname, "../samples/inspect_caliper.viz"), "utf8").replace(/\r\n/g, "\n");
const fields = tol.readLimits(caliper);
assert(fields.map((item) => item.stem).join(",") === "Dia,Width,Height", "caliper limits stay in source order");
assert(fields[0].label === "直徑" && fields[0].min === "80" && fields[0].max === "90", "diameter limits are 80 to 90");
assert(fields[1].min === "130" && fields[1].max === "150", "width limits are 130 to 150");
assert(fields[2].min === "80" && fields[2].max === "100", "height limits are 80 to 100");
const comparison = caliper.split("\n").find((line) => line.includes("Width >= WidthMin"));
const changed = tol.applyLimit(caliper, fields[1].maxLine, "WidthMax", "100");
assert(changed.split("\n")[fields[1].maxLine] === "WidthMax := 100", "the upper width is replaced");
assert(changed.includes(comparison), "the comparison still uses the variable");
assert(tol.applyLimit(caliper, fields[1].maxLine, "WidthMax", "abc") === caliper, "a non-number does not change the program");

const stripe = fs.readFileSync(path.join(__dirname, "../samples/inspect_stripe.viz"), "utf8").replace(/\r\n/g, "\n");
const band = tol.readLimits(stripe);
assert(band.length === 1 && band[0].label === "寬" && band[0].min === "20" && band[0].max === "24", "stripe width limits are 20 to 24");
assert(tol.readLimits("Vizion.read_image (Image, 'a.png')\n").length === 0, "a program without limits has no editor");
const defect = fs.readFileSync(path.join(__dirname, "../samples/inspect_defect.viz"), "utf8").replace(/\r\n/g, "\n");
const gray = tol.readLimits(defect);
assert(gray.map((item) => item.label).join(",") === "暗塊,亮點", "defect gray limits use the same editor");
assert(gray[0].min === "0" && gray[0].max === "40" && gray[1].min === "210" && gray[1].max === "255", "dark is 0 to 40 and bright is 210 to 255");

function memoryStore() {
  const data = {};
  return {
    getItem(key) { return Object.prototype.hasOwnProperty.call(data, key) ? data[key] : null; },
    setItem(key, value) { data[key] = String(value); },
  };
}

const store = memoryStore();
tol.remember(store, "inspect_caliper.viz", changed);
const reopened = tol.restore(store, "samples/inspect_caliper.viz", caliper);
assert(reopened.includes("WidthMax := 100"), "reopening the same template fills the saved upper width");
assert(reopened.includes("DiaMin := 80"), "limits that were not changed stay at the file value");
assert(tol.restore(store, "inspect_stripe.viz", stripe) === stripe, "another template keeps its own limits");
tol.clearStored(store, "inspect_caliper.viz");
assert(tol.restore(store, "inspect_caliper.viz", caliper) === caliper, "clearing the saved limits returns the file");

const chars = "Expect := 'VISDEV 168'\nif (Chars == Expect)\n";
const expects = tol.readExpects(chars);
assert(expects.length === 1 && expects[0].label === "預期字" && expects[0].value === "VISDEV 168", "the expected word is read from the program");
assert(tol.applyExpect(chars, expects[0].line, "Expect", "NOPE").includes("Expect := 'NOPE'"), "a new expected word replaces the assignment");
assert(tol.applyExpect(chars, expects[0].line, "Expect", "A'B") === chars, "a quote is not written into the program");
assert(tol.readLimits(chars).length === 0, "an expected word is not a numeric limit");
const wordStore = memoryStore();
tol.remember(wordStore, "inspect_chars.viz", "Expect := 'NOPE'\n");
assert(tol.restore(wordStore, "samples/inspect_chars.viz", chars).includes("Expect := 'NOPE'"), "reopening the character template fills the saved word");
tol.clearStored(wordStore, "inspect_chars.viz");
assert(tol.restore(wordStore, "inspect_chars.viz", chars) === chars, "clearing the saved word returns VISDEV 168");

const output = "OutHost := 'loopback'\nOutPort := 0\n";
const outs = tol.readOutputs(output);
assert(outs.map((item) => item.label).join(",") === "主機,埠", "output settings name the host and port");
assert(tol.applyOutput(output, outs[0].line, "OutHost", "plc.local").includes("OutHost := 'plc.local'"), "the host can be replaced");
assert(tol.applyOutput(output, outs[1].line, "OutPort", "502").includes("OutPort := 502"), "the port can be replaced");
assert(tol.applyOutput(output, outs[1].line, "OutPort", "70000") === output, "a port above 65535 stays unchanged");
const outStore = memoryStore();
tol.remember(outStore, "inspect_line_out.viz", "OutHost := 'plc.local'\nOutPort := 502\n");
assert(tol.restore(outStore, "samples/inspect_line_out.viz", output).includes("OutPort := 502"), "reopening the line template fills the saved port");
const modes = "OutKind := 'tcp'\nOutHost := 'loopback'\nOutPort := 0\nOutCom := 'loopback'\nOutAddr := 16\n";
const allOut = tol.readOutputs(modes);
assert(tol.visibleOutputs(allOut, "tcp").map((item) => item.name).join(",") === "OutHost,OutPort", "tcp shows host and port");
assert(tol.visibleOutputs(allOut, "serial").map((item) => item.name).join(",") === "OutCom", "serial shows only the port name");
assert(tol.visibleOutputs(allOut, "modbus").map((item) => item.name).join(",") === "OutHost,OutPort,OutAddr", "modbus shows host, port, and address");
assert(tol.applyOutput(modes, 0, "OutKind", "serial").startsWith("OutKind := 'serial'"), "the output kind can be switched");
assert(tol.applyOutput(modes, 0, "OutKind", "file") === modes, "an unknown output kind is refused");

console.log("tolerance ok");
