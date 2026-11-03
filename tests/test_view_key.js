const fs = require("fs");
const vm = require("vm");
const path = require("path");

const source = fs.readFileSync(path.join(__dirname, "..", "app", "static", "view.js"), "utf8");
const window = {};
vm.runInNewContext(source, { window });
const key = window.VisdevView.displayKey;

const base = {
  width: 40,
  height: 20,
  image: "data:image/png;base64,AAAA",
  shapes: [
    { type: "polyline", color: "#00e05a", points: [[1, 2], [8, 2], [8, 9]], object: 0, source: "Region" },
  ],
};

const moved = {
  ...base,
  shapes: [
    { type: "polyline", color: "#00e05a", points: [[4, 6], [12, 6], [12, 14]], object: 0, source: "Region" },
  ],
};

const labeled = {
  ...base,
  shapes: [
    ...base.shapes,
    { type: "text", text: "3", row: 10, col: 4, color: "#ffe14a" },
  ],
};

const sameLengthFill = {
  ...base,
  shapes: [
    ...base.shapes,
    { type: "fill", image: "data:image/png;base64,BBBB" },
  ],
};

const otherFill = {
  ...base,
  shapes: [
    ...base.shapes,
    { type: "fill", image: "data:image/png;base64,CCCC" },
  ],
};

if (!key(base) || key(base) !== key(JSON.parse(JSON.stringify(base)))) {
  throw new Error("same picture must keep the same key");
}
if (key(base) === key(moved)) throw new Error("a moved contour must change the key");
if (key(base) === key(labeled)) throw new Error("a new label must change the key");
if (key(sameLengthFill) === key(otherFill)) throw new Error("a different fill must change the key");
console.log("view key ok");
