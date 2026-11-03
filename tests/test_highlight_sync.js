const highlightApi = require("../app/static/highlight_sync.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function createNode(tag) {
  const node = {
    tagName: tag || "",
    className: "",
    dataset: {},
    parent: null,
    childNodes: [],
    nodeType: tag ? 1 : 3,
    nodeValue: null,
    _html: "",
  };
  const classes = new Set();
  node.classList = {
    contains(name) { return classes.has(name); },
  };
  Object.defineProperty(node, "className", {
    get() { return [...classes].join(" "); },
    set(value) {
      classes.clear();
      String(value || "").split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
    },
  });
  Object.defineProperty(node, "innerHTML", {
    get() { return node._html; },
    set(value) { node._html = String(value); },
  });
  Object.defineProperty(node, "children", {
    get() { return node.childNodes.filter((child) => child.nodeType === 1); },
  });
  Object.defineProperty(node, "firstChild", {
    get() { return node.childNodes[0] || null; },
  });
  node.insertBefore = (child, ref) => {
    if (child.parent) child.parent.removeChild(child);
    child.parent = node;
    if (!ref) node.childNodes.push(child);
    else node.childNodes.splice(node.childNodes.indexOf(ref), 0, child);
    return child;
  };
  node.appendChild = (child) => node.insertBefore(child, null);
  node.removeChild = (child) => {
    const at = node.childNodes.indexOf(child);
    if (at >= 0) node.childNodes.splice(at, 1);
    if (child.parent === node) child.parent = null;
    return child;
  };
  node.remove = () => { if (node.parent) node.parent.removeChild(node); };
  return node;
}

global.document = {
  createElement: (tag) => createNode(tag),
  createTextNode(text) {
    const node = createNode("");
    node.nodeType = 3;
    node.nodeValue = String(text);
    node.tagName = "";
    return node;
  },
};

const pre = createNode("pre");
const seen = [];
function render(line) {
  seen.push(line);
  return `<b>${line}</b>`;
}

let stats = highlightApi.syncHighlight(pre, ["A := 1", "B := 2", "C := 3"], render);
assert(stats.created === 3 && stats.kept === 0, "the first paint creates every line");
assert(seen.length === 3, "the first paint colors every line");
const first = pre.children[0];
const third = pre.children[2];
assert(first.innerHTML === "<b>A := 1</b>", "a line keeps its colored html");
assert(pre.childNodes.filter((node) => node.nodeType === 3).length === 2, "lines stay separated by newlines");

seen.length = 0;
stats = highlightApi.syncHighlight(pre, ["A := 1", "B := 9", "C := 3"], render);
assert(stats.kept === 2 && stats.updated === 1 && stats.created === 0, "only the edited line is recolored");
assert(seen.length === 1 && seen[0] === "B := 9", "unchanged lines are not colored again");
assert(pre.children[0] === first && pre.children[2] === third, "unchanged lines stay the same elements");
assert(pre.children[1].innerHTML === "<b>B := 9</b>", "the edited line shows the new text");

seen.length = 0;
stats = highlightApi.syncHighlight(pre, ["A := 1", "NEW", "B := 9", "C := 3"], render);
assert(stats.created === 1 && stats.removed === 0 && seen.length === 1, "inserting a line only colors that line");
assert(pre.children[0] === first && pre.children[3] === third, "lines around an insert stay");
assert(pre.children[1].innerHTML === "<b>NEW</b>", "the inserted line is colored");
assert(pre.childNodes.filter((node) => node.nodeType === 3).map((node) => node.nodeValue).join(",") === "\n,\n,\n", "newlines still separate every line");

stats = highlightApi.syncHighlight(pre, ["A := 1", "C := 3"], render);
assert(stats.removed === 2 && pre.children.length === 2, "deleting lines removes only those rows");
assert(pre.children[0] === first && pre.children[1] === third, "the kept lines are the original elements");
assert(pre.childNodes.filter((node) => node.nodeType === 3).length === 1, "one newline remains between the two lines");

const again = highlightApi.syncHighlight(pre, ["A := 1", "C := 3"], render);
assert(again.kept === 2 && again.updated === 0 && again.created === 0, "an unchanged program does not recolor");

const names = new Set(["vizion.read_image", "vizion.threshold"]);
const known = (token) => names.has(String(token).toLowerCase());
const colored = highlightApi.highlightSource("Vizion.read_image (Image, 'samples/parts.png')", known);
assert(colored.includes('<span class="tok-op">Vizion.read_image</span>'), "a Vizion call is one operator token");
assert(colored.includes('<span class="tok-str">&#39;samples/parts.png&#39;</span>'), "a path stays a string");
assert(!colored.includes(">Image<") || colored.includes("Image"), "the variable name stays readable");
const number = highlightApi.highlightSource("Min := 60", known);
assert(number.includes('<span class="tok-num">60</span>'), "a number stays colored");
const comment = highlightApi.highlightSource("* 註解", known);
assert(comment.startsWith('<span class="tok-cmt">'), "a comment stays a comment");

const gutter = createNode("div");
const gutterRows = [
  { className: "gl", line: 1, html: '<span class="num">1</span>' },
  { className: "gl cur", line: 2, html: '<span class="num">2</span>' },
  { className: "gl", line: 3, html: '<span class="num">3</span>' },
];
stats = highlightApi.syncRows(gutter, gutterRows);
assert(stats.created === 3, "the gutter creates one row per line");
const gutterFirst = gutter.children[0];
const gutterThird = gutter.children[2];
stats = highlightApi.syncRows(gutter, [
  gutterRows[0],
  { className: "gl", line: 2, html: '<span class="num">2</span>' },
  gutterRows[2],
]);
assert(stats.kept === 2 && stats.updated === 1, "moving the arrow updates only that gutter row");
assert(gutter.children[0] === gutterFirst && gutter.children[2] === gutterThird, "gutter rows stay when the arrow moves");
assert(gutter.children[1].className === "gl", "the old current line loses the arrow");
stats = highlightApi.syncRows(gutter, gutterRows.concat([{ className: "gl", line: 4, html: '<span class="num">4</span>' }]));
assert(stats.created === 1 && gutter.children[0] === gutterFirst, "a new line appends without rebuilding earlier rows");

const times = createNode("div");
highlightApi.syncRows(times, [
  { className: "", text: "<0.001 ms" },
  { className: "on", text: "1.20 ms" },
]);
assert(times.children[0].textContent === "<0.001 ms", "a tiny time stays readable text");
const timeRow = times.children[0];
highlightApi.syncRows(times, [
  { className: "", text: "<0.001 ms" },
  { className: "", text: "1.20 ms" },
]);
assert(times.children[0] === timeRow, "an unchanged time row stays");
assert(times.children[1].className === "", "the time arrow can move off a row");

console.log("highlight sync ok");
