const geom = require("../app/static/geom.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

const square = [[0, 0], [10, 0], [10, 10], [0, 10]];
assert(geom.pointInPoly(5, 5, square), "center of a square is inside");
assert(!geom.pointInPoly(15, 5, square), "a point to the right is outside");
assert(geom.pointInPoly(5, 0, square) === true || geom.pointInPoly(5, 0, square) === false, "a horizontal edge does not throw");

const dist = geom.lineDistance(5, 5, [[0, 0], [10, 0]]);
assert(Math.abs(dist - 5) < 1e-9, "distance to a horizontal segment is 5");

const region = {
  type: "polyline",
  kind: "region",
  object: 0,
  source: "Block",
  points: square,
};
const line = {
  type: "polyline",
  kind: "xld",
  object: 1,
  source: "Edge",
  points: [[0, 20], [10, 20]],
};
const nearLine = geom.pickShape([region, line], 5, 16, 1);
assert(nearLine && nearLine.kind === "xld" && nearLine.name === "Edge", "a nearby contour wins over a region");
const inRegion = geom.pickShape([region, line], 5, 5, 1);
assert(inRegion && inRegion.kind === "region" && inRegion.index === 0, "a click inside the region selects it");
assert(geom.pickShape([region, line], 5, 40, 1) === null, "empty space selects nothing");
assert(geom.pickShape([line], 5, 40, 1) === null, "far from a contour at scale 1 misses");
const wide = geom.pickShape([line], 5, 40, 0.4);
assert(wide && wide.kind === "xld", "zoomed out, the same contour is still within the pick radius");

const outer = {
  type: "polyline",
  kind: "xld",
  object: 0,
  group: 0,
  source: "Ring",
  points: [[0, 0], [20, 0], [20, 20], [0, 20], [0, 0]],
};
const inner = {
  type: "polyline",
  kind: "xld",
  object: 1,
  group: 1,
  source: "Hole",
  points: [[6, 6], [8, 6], [8, 8], [6, 8], [6, 6]],
};
const hole = geom.pickShape([outer, inner], 7, 7, 1);
assert(hole && hole.name === "Hole" && hole.index === 1, "the smaller closed contour wins when the click is inside both");

const fitted = geom.fitView(220, 140, 100, 50, 18);
assert(Math.abs(fitted.scale - 1.84) < 1e-12, "fit uses the tighter axis");
assert(Math.abs(fitted.x - 18) < 1e-12 && Math.abs(fitted.y - 24) < 1e-12, "fit centers the image");
const tiny = geom.fitView(20, 20, 1000, 1000, 18);
assert(tiny.scale === 0.02, "fit does not shrink below 0.02");

const one = geom.oneToOne(200, 100, 80, 40);
assert(one.scale === 1 && one.x === 60 && one.y === 30, "1:1 puts the image center on the view center");

const start = { scale: 2, x: 10, y: 20 };
const anchor = geom.imagePoint(100, 80, start);
const zoomed = geom.zoomAt(start, 100, 80, 1.25);
const stayed = geom.imagePoint(100, 80, zoomed);
assert(Math.abs(anchor.col - stayed.col) < 1e-9 && Math.abs(anchor.row - stayed.row) < 1e-9, "zoom keeps the point under the cursor");
assert(geom.zoomAt({ scale: 40, x: 0, y: 0 }, 0, 0, 2).scale === 48, "zoom stops at 48");
assert(geom.zoomAt({ scale: 0.06, x: 0, y: 0 }, 0, 0, 0.5).scale === 0.05, "zoom stops at 0.05");

const framed = geom.zoomRect(200, 200, 30, 80, 10, 40, 8);
const center = geom.imagePoint(100, 100, framed);
assert(Math.abs(center.col - 20) < 1e-9 && Math.abs(center.row - 60) < 1e-9, "a zoom rectangle is centered in the view");

const panned = geom.panBy(10, 20, 8, -4);
assert(panned.x === 18 && panned.y === 16, "pan adds the pointer movement");
const before = geom.imagePoint(50, 50, { scale: 2, x: 10, y: 20 });
const after = geom.imagePoint(50, 50, { scale: 2, x: panned.x, y: panned.y });
assert(Math.abs(before.col - 8 / 2 - after.col) < 1e-9, "panning right moves the image left under the same pixel");
assert(Math.abs(before.row - (-4) / 2 - after.row) < 1e-9, "panning up moves the image down under the same pixel");

const box = geom.draftBox({ col: 10, row: 30 }, { col: 2, row: 8 });
assert(box.col1 === 2 && box.col2 === 10 && box.row1 === 8 && box.row2 === 30, "a dragged rectangle sorts its corners");
assert(box.width === 8 && box.height === 22, "rectangle size follows the sorted corners");

const cornerTools = geom.geometryTools(["Row1", "Column1", "Row2", "Column2", "Row", "Column", "Length1", "Length2"]);
assert(cornerTools[0] === "rect" && cornerTools.length === 1, "corner fields choose the corner rectangle");
const centerTools = geom.geometryTools(["Row", "Column", "Length1", "Length2", "Radius"]);
assert(centerTools.indexOf("rect") !== -1 && centerTools.indexOf("circle") !== -1, "a center rectangle can also draw a circle");

const circle = geom.shapeNumbers("circle", { col: 4, row: 5 }, { col: 7, row: 9 }, "axis");
assert(circle.Row === 5 && circle.Column === 4 && Math.abs(circle.Radius - 5) < 1e-12, "circle radius is the drag length");
const centered = geom.shapeNumbers("rect", { col: 0, row: 0 }, { col: 10, row: 4 }, "axis");
assert(centered.Row === 2 && centered.Column === 5 && centered.Phi === 0, "a center rectangle sits on the box middle");
assert(centered.Length1 === 5 && centered.Length2 === 2, "half widths are half the box");
const ring = geom.shapeNumbers("ellipse", { col: 0, row: 0 }, { col: 10, row: 4 }, "inner");
assert(ring.Radius1 === 2 && ring.Radius2 === 5, "an inner radius uses the shorter axis first");

console.log("canvas pick ok");
