const historyApi = require("../app/static/editor_history.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

let clock = 0;
const history = historyApi.createEditorHistory({
  gap: 800,
  now: function () { return clock; },
});
history.reset("A", 1, 1);

clock = 1000;
history.commit("AB", 2, 2, "type");
clock = 1100;
history.commit("ABC", 3, 3, "type");
clock = 1200;
history.commit("ABCD", 4, 4, "type");
let back = history.undo();
assert(back.value === "A", "a typing burst undoes as one step");
assert(!history.canUndo(), "nothing remains before the burst");
let again = history.redo();
assert(again.value === "ABCD", "redo restores the burst");

clock = 3000;
history.commit("ABCD\nVizion.threshold (Image, Region, 60, 255)", 40, 40, "checkpoint");
back = history.undo();
assert(back.value === "ABCD", "an inserted operator is its own undo step");
back = history.undo();
assert(back.value === "A", "the next undo still removes the typing burst");

clock = 4000;
history.reset("program", 7, 7);
history.commit("program!", 8, 8, "type");
assert(history.undo().value === "program", "reset drops the previous file");
assert(!history.canUndo(), "reset leaves no older step");

clock = 5000;
history.reset("", 0, 0);
history.beginComposition();
history.commit("你", 1, 1, "type");
history.commit("你好", 2, 2, "type");
history.endComposition("你好", 2, 2);
back = history.undo();
assert(back.value === "", "one composition is one undo step");
assert(!history.canUndo(), "composition does not leave intermediate steps");

console.log("editor history ok");
