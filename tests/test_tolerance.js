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

console.log("tolerance ok");
