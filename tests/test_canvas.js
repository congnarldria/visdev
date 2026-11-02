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

console.log("canvas pick ok");
