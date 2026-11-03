const varsApi = require("../app/static/var_list.js");

function assert(cond, message) {
  if (!cond) throw new Error(message);
}

function createNode(tag) {
  const node = {
    tagName: tag,
    dataset: {},
    title: "",
    parent: null,
    children: [],
    nodeType: 1,
    nodeValue: null,
    _attrs: {},
    _text: "",
  };
  const classes = new Set();
  function paintClass() {
    node._className = [...classes].join(" ");
  }
  node._className = "";
  node.classList = {
    contains(name) { return classes.has(name); },
    add(...names) { names.forEach((name) => classes.add(name)); paintClass(); },
    remove(...names) { names.forEach((name) => classes.delete(name)); paintClass(); },
    toggle(name, force) {
      const on = force == null ? !classes.has(name) : !!force;
      if (on) classes.add(name);
      else classes.delete(name);
      paintClass();
      return on;
    },
  };
  Object.defineProperty(node, "className", {
    get() { return node._className; },
    set(value) {
      classes.clear();
      String(value || "").split(/\s+/).filter(Boolean).forEach((name) => classes.add(name));
      paintClass();
    },
  });
  Object.defineProperty(node, "textContent", {
    get() {
      if (node.nodeType === 3) return node.nodeValue || "";
      if (!node.children.length) return node._text;
      return node.children.map((child) => child.textContent).join("");
    },
    set(value) {
      node._text = String(value);
      node.children = [];
    },
  });
  Object.defineProperty(node, "firstChild", {
    get() { return node.children[0] || null; },
  });
  Object.defineProperty(node, "previousSibling", {
    get() {
      if (!node.parent) return null;
      const at = node.parent.children.indexOf(node);
      return at > 0 ? node.parent.children[at - 1] : null;
    },
  });
  node.setAttribute = (key, value) => { node._attrs[key] = String(value); };
  node.getAttribute = (key) => (Object.prototype.hasOwnProperty.call(node._attrs, key) ? node._attrs[key] : null);
  Object.defineProperty(node, "src", {
    get() { return node.getAttribute("src") || ""; },
    set(value) { node.setAttribute("src", value); },
  });
  node.appendChild = (child) => node.insertBefore(child, null);
  node.insertBefore = (child, ref) => {
    if (child.parent) child.parent.removeChild(child);
    child.parent = node;
    if (!ref) node.children.push(child);
    else node.children.splice(node.children.indexOf(ref), 0, child);
    return child;
  };
  node.removeChild = (child) => {
    const at = node.children.indexOf(child);
    if (at >= 0) node.children.splice(at, 1);
    if (child.parent === node) child.parent = null;
    return child;
  };
  node.remove = () => { if (node.parent) node.parent.removeChild(node); };
  node.querySelector = (selector) => {
    const all = [];
    function walk(current) {
      (current.children || []).forEach((child) => {
        if (child.nodeType === 1) all.push(child);
        if (child.children) walk(child);
      });
    }
    walk(node);
    if (selector === "img") return all.find((item) => item.tagName === "img") || null;
    if (selector === "small") return all.find((item) => item.tagName === "small") || null;
    if (selector === ".vn") return all.find((item) => item.classList.contains("vn")) || null;
    if (selector === ".vm") return all.find((item) => item.classList.contains("vm")) || null;
    return null;
  };
  return node;
}

global.document = {
  createElement: createNode,
  createTextNode(text) {
    const node = {
      nodeType: 3,
      nodeValue: String(text),
      textContent: String(text),
      children: [],
      parent: null,
      tagName: "",
    };
    Object.defineProperty(node, "previousSibling", {
      get() {
        if (!node.parent) return null;
        const at = node.parent.children.indexOf(node);
        return at > 0 ? node.parent.children[at - 1] : null;
      },
    });
    node.remove = () => { if (node.parent) node.parent.removeChild(node); };
    return node;
  },
};

const iconic = [
  { name: "Image", kind: "image", role: "image", info: "80×60  byte", thumb: "data:image/gif;base64,aaa" },
];
const control = [
  { name: "A", type: "integer", text: "1", full: "1" },
  { name: "B", type: "integer", text: "2", full: "2" },
];

const images = createNode("div");
const numbers = createNode("div");
let first = varsApi.syncVarList(images, iconic, "iconic", null, "還沒有圖像變數");
varsApi.syncVarList(numbers, control, "control", null, "還沒有控制變數");
assert(first.created === 1 && first.kept === 0, "the first paint creates the image row");
const imageRow = images.children[0];
const imageTag = imageRow.querySelector("img");
const rowA = numbers.children[0];
const rowB = numbers.children[1];
assert(imageRow.querySelector(".vn").textContent === "Image", "the image row shows its name");
assert(imageTag.getAttribute("src") === iconic[0].thumb, "the thumb is the data url");
assert(rowA.querySelector(".vm").textContent === "1", "A shows 1");

const changed = control.map((item) => ({ ...item }));
changed[1] = { ...changed[1], text: "9", full: "9" };
const second = varsApi.syncVarList(numbers, changed, "control", "A", "還沒有控制變數");
assert(second.updated === 1 && second.kept === 1 && second.created === 0, "only the changed number row is rewritten");
assert(numbers.children[0] === rowA, "A stays the same element");
assert(numbers.children[1] === rowB, "B stays the same element");
assert(rowA.classList.contains("on"), "the selected row is marked");
assert(!rowB.classList.contains("on"), "the other row is not selected");
assert(rowB.querySelector(".vm").textContent === "9", "B shows the new value");
assert(imageRow.querySelector("img") === imageTag, "updating numbers does not touch the image");

const third = varsApi.syncVarList(images, iconic, "iconic", null, "還沒有圖像變數");
assert(third.kept === 1 && third.updated === 0 && third.created === 0, "an unchanged image row is left alone");
assert(images.children[0] === imageRow, "the image row element is the same");
assert(images.children[0].querySelector("img") === imageTag, "the thumb element is the same");

const reordered = varsApi.syncVarList(numbers, [changed[1], changed[0]], "control", "A", "還沒有控制變數");
assert(reordered.created === 0 && reordered.removed === 0, "reordering reuses both rows");
assert(numbers.children[0] === rowB && numbers.children[1] === rowA, "the rows move without being recreated");

const trimmed = varsApi.syncVarList(numbers, [changed[0]], "control", null, "還沒有控制變數");
assert(trimmed.removed === 1 && trimmed.kept === 1, "a removed variable drops only that row");
assert(numbers.children.length === 1 && numbers.children[0] === rowA, "A remains");
assert(!rowA.classList.contains("on"), "clearing the selection removes the mark");

const empty = varsApi.syncVarList(images, [], "iconic", null, "還沒有圖像變數");
assert(empty.created === 1, "an empty list shows the hint");
assert(images.children[0].classList.contains("hint"), "the hint is a hint row");
const hint = images.children[0];
const again = varsApi.syncVarList(images, [], "iconic", null, "還沒有圖像變數");
assert(again.created === 0 && images.children[0] === hint, "the empty hint is not rebuilt");

const back = varsApi.syncVarList(images, iconic, "iconic", null, "還沒有圖像變數");
assert(back.created === 1 && images.children[0].querySelector("img").getAttribute("src") === iconic[0].thumb, "a returning image builds a new row");

const log = createNode("pre");
let logged = varsApi.syncLog(log, ["開始", "A := 1"]);
assert(logged.kept === 0 && logged.appended === 2, "the first log creates both lines");
const firstLine = [...log.children].find((node) => node.tagName === "span");
const secondLine = [...log.children].filter((node) => node.tagName === "span")[1];
logged = varsApi.syncLog(log, ["開始", "A := 1", "B := 2"]);
assert(logged.kept === 2 && logged.appended === 1, "a new log line is appended");
assert([...log.children].find((node) => node.tagName === "span") === firstLine, "the first log line stays");
assert([...log.children].filter((node) => node.tagName === "span")[1] === secondLine, "the second log line stays");
assert([...log.children].filter((node) => node.tagName === "span")[2].textContent === "B := 2", "the third line is the new text");
logged = varsApi.syncLog(log, ["開始", "錯誤：讀不到", "B := 2"]);
assert(logged.kept === 1 && logged.appended === 2, "a changed middle line rewrites the tail");
const rewritten = [...log.children].filter((node) => node.tagName === "span");
assert(rewritten[0] === firstLine, "the unchanged head stays");
assert(rewritten[1].classList.contains("bad"), "an error line is marked");
assert(rewritten[1].textContent === "錯誤：讀不到", "the error text is shown");
logged = varsApi.syncLog(log, []);
assert(logged.kept === 0 && logged.appended === 0 && log.children.length === 0, "clearing the log removes every line");

const problems = createNode("tbody");
const labels = { error: "錯誤", warning: "警告", message: "訊息" };
let listed = varsApi.syncProblems(problems, [], labels);
assert(listed.created === 1 && problems.children[0].dataset.empty === "1", "an empty problem list shows the placeholder");
const placeholder = problems.children[0];
assert(varsApi.syncProblems(problems, [], labels).created === 0 && problems.children[0] === placeholder, "the placeholder stays");
const warningA = { sev: "warning", code: "VIZ100", text: "變數 A 指定了但沒有被使用", file: "程式", line: 1 };
const warningB = { sev: "warning", code: "VIZ100", text: "變數 B 指定了但沒有被使用", file: "程式", line: 2 };
listed = varsApi.syncProblems(problems, [warningA], labels);
assert(listed.created === 1 && problems.children.length === 1, "the first warning replaces the placeholder");
const problemA = problems.children[0];
assert(problemA.dataset.line === "1", "the warning remembers its line");
assert(problemA.children[2].textContent === warningA.text, "the warning text is shown");
listed = varsApi.syncProblems(problems, [warningA, warningB], labels);
assert(listed.kept === 1 && listed.created === 1, "adding a warning keeps the older row");
assert(problems.children[0] === problemA, "the first warning element stays");
assert(problems.children[1].dataset.line === "2", "the new warning is second");
const problemB = problems.children[1];
listed = varsApi.syncProblems(problems, [warningB, warningA], labels);
assert(listed.created === 0 && listed.removed === 0, "reordering warnings reuses both rows");
assert(problems.children[0] === problemB && problems.children[1] === problemA, "warning rows move without being recreated");

console.log("var list ok");
