const pick = require("../app/static/pick_sync.js");
const geom = require("../app/static/geom.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

const hit = { name: "Region", index: 2, kind: "region" };
const message = pick.pack("main", hit, "s1");
message.at = 10;
const selected = pick.graphicsSelection(message);
assert(selected.name === "Region" && selected.index === 2 && selected.kind === "region", "a graphics pick keeps the object");
assert(pick.sameSelection(selected, { name: "Region", index: 2, kind: "region" }), "the same object matches on both windows");

const cleared = pick.pack("gfx", null, "s1");
cleared.at = 11;
assert(pick.graphicsSelection(cleared) === null, "a clear removes the selection");
assert(pick.graphicsSelection({ type: "pick", name: "__preview__", index: 0, kind: "region", at: 12 }) === null, "a threshold preview does not select the graphics window");

const xld = pick.graphicsSelection({ type: "pick", name: "Edge", index: 1, kind: "xld", at: 13 });
assert(xld.group === 1, "an xld pick uses its group so both windows stroke the same contour");

const gfx = pick.createWatch("gfx");
assert(gfx.accept(message) && gfx.accept(message) === null, "the same pick is applied once");
const main = pick.createWatch("main");
const echo = pick.pack("main", hit, "s1");
echo.at = 20;
assert(main.accept(echo) === null, "a window ignores its own pick");
const remote = pick.pack("gfx", { name: "Blob", index: 0, kind: "region" }, "s1");
remote.at = 21;
const taken = main.accept(remote);
assert(taken.selection.name === "Blob" && taken.selection.index === 0, "a pick from the other window is kept");
const again = pick.pack("gfx", { name: "Blob", index: 0, kind: "region" }, "s1");
again.at = 22;
assert(pick.sameSelection(taken.selection, main.accept(again).selection), "a repeated object stays the same selection");

assert(geom.shouldRefit(true, 640, 480, 640, 480) === false, "a user zoom survives the same picture");
assert(geom.shouldRefit(true, 640, 480, 800, 600) === true, "a new image size fits again");
assert(geom.shouldRefit(true, 0, 0, 640, 480) === false, "a frame placed before the picture is kept");
assert(geom.shouldRefit(false, 640, 480, 640, 480) === true, "the first view still fits the window");
assert(geom.shouldRefit(true, 640, 480, 0, 0) === false, "an empty frame does not refit");
const framed = geom.frameAt(200, 100, 2, 10, 20);
assert(framed.scale === 2 && framed.x === 80 && framed.y === 10, "frame puts the image point at the view center");
const back = geom.imagePoint(100, 50, framed);
assert(Math.abs(back.col - 10) < 1e-9 && Math.abs(back.row - 20) < 1e-9, "the center stays on that image point");

const text = pick.summary({ area: 120, row: 10.04, column: 20.04 });
assert(text === "面積 120  中心 10.0, 20.0", "area and center use one decimal");
const packed = pick.pack("main", { name: "Blob", index: 0, kind: "region", summary: text }, "s1");
assert(pick.graphicsSelection(packed).summary === text, "the summary travels with the pick");
assert(pick.summary({}) === "", "missing area leaves the label without numbers");

console.log("pick sync ok");
