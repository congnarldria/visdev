const calls = require("../app/static/operator_call.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function same(left, right, message) {
  assert(JSON.stringify(left) === JSON.stringify(right), message + ": " + JSON.stringify(left));
}

const threshold = {
  name: "Vizion.threshold",
  example: "Vizion.threshold (Image, Region, 128, 255)",
  params: [
    { name: "Image", role: "圖像輸入" },
    { name: "Region", role: "圖像輸出" },
    { name: "MinGray", role: "控制輸入" },
    { name: "MaxGray", role: "控制輸入" },
  ],
};

const find = {
  name: "Vizion.find_shape_model",
  example: "Vizion.find_shape_model (Image, ModelID, rad(-180), rad(360), 0.6, 0, 0.5, 'interpolation', 0, 0.8, Row, Column, Angle, Score)",
  params: [
    { name: "Image", role: "圖像輸入" },
    { name: "ModelID", role: "控制輸入" },
    { name: "AngleStart", role: "控制輸入" },
    { name: "AngleExtent", role: "控制輸入" },
    { name: "MinScore", role: "控制輸入" },
    { name: "NumMatches", role: "控制輸入" },
    { name: "MaxOverlap", role: "控制輸入" },
    { name: "SubPixel", role: "控制輸入" },
    { name: "NumLevels", role: "控制輸入" },
    { name: "Greediness", role: "控制輸入" },
    { name: "Row", role: "控制輸出" },
    { name: "Column", role: "控制輸出" },
    { name: "Angle", role: "控制輸出" },
    { name: "Score", role: "控制輸出" },
  ],
};

const thresholdValues = calls.callValues(threshold);
const thresholdCall = calls.formatCall(threshold.name, thresholdValues);
const thresholdParsed = calls.parseLine("  " + thresholdCall);
same(thresholdParsed.args, thresholdValues, "threshold call round-trips");
same(calls.callValues(threshold, thresholdParsed.args), thresholdValues, "parsed threshold fills the form");

const findValues = calls.callValues(find);
const findCall = calls.formatCall(find.name, findValues);
const findParsed = calls.parseLine(findCall);
assert(findParsed.name === "Vizion.find_shape_model", "keeps the operator name");
same(findParsed.args, findValues, "shape match call round-trips");
assert(findParsed.args[2] === "rad(-180)", "nested call stays one argument");
assert(findParsed.args[7] === "'interpolation'", "quoted choice stays one argument");

const quoted = calls.parseLine("Vizion.foo (Image, 'a, b', rad(1))");
same(quoted.args, ["Image", "'a, b'", "rad(1)"], "comma inside quotes is not a separator");

const shortFind = calls.callValues(find, ["Photo", "Model", "rad(-180)"]);
assert(shortFind[0] === "Photo", "a short call keeps the image name");
assert(shortFind[1] === "Model", "a short call keeps the model name");
assert(shortFind[2] === "rad(-180)", "a short call keeps the angle in place");
assert(shortFind[13] === "Score", "missing outputs are not taken from earlier arguments");

const saved = calls.memoryRecord(threshold, ["Photo", "Dark", "40", "200"]);
same(saved, { Image: "Photo", Region: "Dark", MinGray: "40", MaxGray: "200" }, "memory stores every role");
const inserted = calls.dialogValues(threshold, { memory: saved });
assert(inserted.label === "插入", "a new call is an insert");
same(inserted.values, ["Photo", "Dark", "40", "200"], "the next insert uses the last names and inputs");
assert(inserted.call === "Vizion.threshold (Photo, Dark, 40, 200)", "preview matches the call that will be written");

const partial = calls.valuesWithMemory(threshold, { MinGray: "10" });
same(partial, ["Image", "Region", "10", "255"], "a remembered input leaves the other fields on the example");

const edited = calls.dialogValues(threshold, {
  lineIndex: 1,
  preset: calls.callValues(threshold, ["Image", "Region", "128", "255"]),
  memory: saved,
});
assert(edited.label === "更新", "an existing line is an update");
same(edited.values, ["Image", "Region", "128", "255"], "the line wins over memory");

const source = "* 註解\n  Vizion.threshold (Image, Region, 128, 255)\nVizion.read_image (Image, 'samples/parts.png')";
const replaced = calls.replaceLine(source, 1, "Vizion.threshold (Photo, Dark, 40, 200)");
same(replaced.text.split("\n"), [
  "* 註解",
  "  Vizion.threshold (Photo, Dark, 40, 200)",
  "Vizion.read_image (Image, 'samples/parts.png')",
], "update rewrites that line and keeps the indent");
assert(!replaced.text.includes("* 這"), "updating a line does not insert a note");

console.log("operator call ok");
