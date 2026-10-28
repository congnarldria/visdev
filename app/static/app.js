const code = document.getElementById("code");
const hl = document.getElementById("hl");
const gutter = document.getElementById("gutter");
const currentBar = document.getElementById("currentBar");
const errBar = document.getElementById("errBar");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const gfxEmpty = document.getElementById("gfxEmpty");
const gfxBanner = document.getElementById("gfxBanner");
const gfxTitle = document.getElementById("gfxTitle");
const iconicList = document.getElementById("iconicList");
const controlList = document.getElementById("controlList");
const iconicCount = document.getElementById("iconicCount");
const varDetail = document.getElementById("varDetail");
const logBox = document.getElementById("logBox");
const opList = document.getElementById("opList");
const opHelp = document.getElementById("opHelp");
const opSearch = document.getElementById("opSearch");
const sampleMenu = document.getElementById("sampleMenu");
const stState = document.getElementById("stState");
const stLine = document.getElementById("stLine");
const stZoom = document.getElementById("stZoom");
const stPix = document.getElementById("stPix");

const LINE = 20;
const PAD = 8;
let operators = [];
let operatorNames = new Set();
let sessionId = null;
let server = { line: null, done: false, error: null, errorLine: null, iconic: [], control: [], display: null, log: [], paused: null };
let breakpoints = new Set();
let selectedVar = null;
let inflight = false;
let cam = { x: 0, y: 0, scale: 1, user: false, fitW: 0, fitH: 0 };
let disp = null;
let baseImage = null;
let loadToken = 0;
let dprScale = 1;
let mouse = null;
let featureLines = [];
let pickedObject = null;
const probe = document.createElement("canvas");
const probeCtx = probe.getContext("2d", { willReadFrequently: true });

function esc(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[ch]));
}

function highlightLine(line) {
  if (/^\s*(?:\*|\/\/)/.test(line)) return `<span class="tok-cmt">${esc(line)}</span>`;
  const re = /('[^']*'|"[^"]*"|\b(?:if|else|endif|for|endfor|while|endwhile)\b|\b\d+(?:\.\d+)?\b|[A-Za-z_]\w*|\s+|.)/g;
  let html = "";
  let match;
  while ((match = re.exec(line))) {
    const token = match[0];
    let cls = "";
    if (token[0] === "'" || token[0] === '"') cls = "tok-str";
    else if (/^\d/.test(token)) cls = "tok-num";
    else if (/^(if|else|endif|for|endfor|while|endwhile)$/i.test(token)) cls = "tok-key";
    else if (operatorNames.has(token.toLowerCase())) cls = "tok-op";
    html += cls ? `<span class="${cls}">${esc(token)}</span>` : esc(token);
  }
  return html || " ";
}

function renderHighlight() {
  const src = code.value.endsWith("\n") ? code.value + " " : code.value;
  hl.innerHTML = src.split("\n").map(highlightLine).join("\n");
}

function renderGutter() {
  const count = Math.max(1, code.value.split("\n").length);
  let html = "";
  for (let line = 1; line <= count; line += 1) {
    const cls = ["gl"];
    if (breakpoints.has(line)) cls.push("bp");
    if (server.line === line) cls.push("cur");
    if (server.errorLine === line) cls.push("err");
    html += `<div class="${cls.join(" ")}" data-line="${line}"><span class="pc"></span><span class="num">${line}</span></div>`;
  }
  gutter.innerHTML = html;
}

function placeBars() {
  const topFor = (line) => PAD + (line - 1) * LINE - code.scrollTop;
  if (server.line) {
    currentBar.style.display = "block";
    currentBar.style.top = `${topFor(server.line)}px`;
  } else {
    currentBar.style.display = "none";
  }
  if (server.errorLine) {
    errBar.style.display = "block";
    errBar.style.top = `${topFor(server.errorLine)}px`;
  } else {
    errBar.style.display = "none";
  }
}

function renderEditor() {
  renderHighlight();
  renderGutter();
  placeBars();
}

function renderVars() {
  iconicCount.textContent = server.iconic.length ? String(server.iconic.length) : "";
  iconicList.innerHTML = server.iconic.map((item) => `
    <button class="var ${selectedVar === item.name ? "on" : ""}" data-name="${esc(item.name)}" data-kind="iconic">
      <img alt="" src="${item.thumb || ""}">
      <span class="vn">${esc(item.name)}</span>
      <span class="vm">${esc(item.kind)} · ${esc(item.info)}</span>
    </button>`).join("") || `<div class="hint">還沒有圖像變數</div>`;
  controlList.innerHTML = server.control.map((item) => `
    <button class="var ctrl ${selectedVar === item.name ? "on" : ""}" data-name="${esc(item.name)}" data-kind="control">
      <span class="vn">${esc(item.name)}</span>
      <span class="vt">${esc(item.type)}</span>
      <span class="vm">${esc(item.text)}</span>
    </button>`).join("") || `<div class="hint">還沒有控制變數</div>`;
}

function renderLog() {
  const lines = server.log || [];
  logBox.innerHTML = lines.map((line) => {
    const cls = String(line).startsWith("錯誤") ? "bad" : "";
    return `<span class="${cls}">${esc(line)}</span>`;
  }).join("\n");
  logBox.scrollTop = logBox.scrollHeight;
}

function setStatus() {
  stState.className = "";
  if (server.error) {
    stState.textContent = "錯誤";
    stState.className = "bad";
  } else if (server.paused === "breakpoint") {
    stState.textContent = "中斷";
    stState.className = "pause";
  } else if (server.paused === "stop") {
    stState.textContent = "暫停";
    stState.className = "pause";
  } else if (server.paused === "stopped") {
    stState.textContent = "已停止";
    stState.className = "pause";
  } else if (server.done) {
    stState.textContent = "完成";
    stState.className = "ok";
  } else if (server.line) {
    stState.textContent = "就緒";
  } else {
    stState.textContent = "就緒";
  }
  stLine.textContent = server.line ? `下一行 ${server.line}` : (server.done ? "程式結束" : "");
}

function applyState(data) {
  previewImg = null;
  previewToken += 1;
  featureLines = [];
  sessionId = data.id;
  server = data;
  gfxBanner.hidden = true;
  renderEditor();
  renderVars();
  renderLog();
  setStatus();
  if (data.display) loadDisplay(data.display);
  else clearGraphics();
  renderHisto(data.histogram || null);
  if (data.error) {
    showTab("log");
    showErrorDialog(data.error, data.errorLine);
  }
}

function errorText(message) {
  const err = message?.detail;
  if (typeof err === "string") return { text: err, line: message?.line || null };
  if (Array.isArray(err)) {
    const text = err.map((item) => item?.msg || String(item)).join("\n");
    return { text: text || "請求失敗", line: null };
  }
  if (err && typeof err === "object") return { text: err.error || "請求失敗", line: err.line || null };
  return { text: message?.error || "請求失敗", line: message?.line || null };
}

function showErrorDialog(text, line) {
  const box = document.getElementById("errDialog");
  document.getElementById("errText").textContent = text || "發生錯誤";
  document.getElementById("errWhere").textContent = line ? `第 ${line} 行` : "";
  box.hidden = false;
  document.getElementById("errOk").focus();
}

function closeErrorDialog() {
  document.getElementById("errDialog").hidden = true;
}

function showDetail(message) {
  const parsed = errorText(message);
  server.error = parsed.text;
  server.errorLine = parsed.line;
  server.log = (server.log || []).concat(parsed.text);
  renderEditor();
  renderLog();
  setStatus();
  showTab("log");
  showErrorDialog(parsed.text, parsed.line);
}

async function exec(action, line) {
  if (inflight) return;
  inflight = true;
  document.getElementById("btnRun").disabled = action === "run";
  try {
    const res = await fetch("/api/exec", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: sessionId,
        source: code.value,
        breakpoints: [...breakpoints],
        action,
        trace: document.getElementById("trace").checked,
        regionColor: regionColorValue(),
        line: line ?? null,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      showDetail(data);
      return;
    }
    applyState(data);
  } catch (err) {
    showDetail({ error: "無法連上 VisDev：" + err.message });
  } finally {
    inflight = false;
    document.getElementById("btnRun").disabled = false;
  }
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.toggle("on", tab.dataset.tab === name));
  document.querySelectorAll(".tabpage").forEach((page) => page.classList.toggle("on", page.dataset.page === name));
}

function renderOps() {
  const query = opSearch.value.trim().toLowerCase();
  const visible = operators.filter((op) => !query || (op.name + op.short + op.detail).toLowerCase().includes(query));
  let html = "";
  let group = "";
  for (const op of visible) {
    if (op.group !== group) {
      group = op.group;
      html += `<div class="op-group">${esc(group)}</div>`;
    }
    html += `<button class="op" data-name="${esc(op.name)}" type="button"><b>${esc(op.name)}</b><span>${esc(op.short)}</span></button>`;
  }
  opList.innerHTML = html || `<div class="hint" style="padding:10px">沒有符合的算子</div>`;
}

function showOp(name) {
  const op = operators.find((item) => item.name === name);
  if (!op) return;
  document.querySelectorAll(".op").forEach((node) => node.classList.toggle("on", node.dataset.name === name));
  const rows = op.params.map((param) => `<tr><td>${esc(param.name)}</td><td class="role">${esc(param.role)}</td><td>${esc(param.desc)}</td></tr>`).join("");
  opHelp.innerHTML = `
    <h2>${esc(op.name)}</h2>
    <p class="lead">${esc(op.short)}</p>
    <p>${esc(op.detail)}</p>
    <div class="hint">原型</div>
    <pre>${esc(op.prototype)}</pre>
    <div class="hint">程式呼叫</div>
    <pre>${esc(op.example)}</pre>
    ${rows ? `<table class="params"><tr><th>參數</th><th>角色</th><th>說明</th></tr>${rows}</table>` : ""}
    <button id="btnInsert" type="button">插入範本</button>`;
  document.getElementById("btnInsert").onclick = () => insertAtCursor(op.example);
}

function insertAtCursor(text) {
  const pos = code.selectionStart;
  const start = code.value.slice(0, pos).lastIndexOf("\n") + 1;
  code.value = code.value.slice(0, start) + text + "\n" + code.value.slice(start);
  const caret = start + text.length + 1;
  code.selectionStart = code.selectionEnd = caret;
  renderEditor();
  code.focus();
}

const SENSE_KEYS = [
  { name: "if", insert: "if ()", caret: 4, detail: "條件成立才執行，直到 endif" },
  { name: "else", insert: "else", detail: "條件不成立時的分支" },
  { name: "endif", insert: "endif", detail: "結束 if" },
  { name: "for", insert: "for Index := 1 to 10 by 1", detail: "從起點數到終點，步進 by" },
  { name: "endfor", insert: "endfor", detail: "結束 for" },
  { name: "while", insert: "while ()", caret: 7, detail: "條件成立就重複，直到 endwhile" },
  { name: "endwhile", insert: "endwhile", detail: "結束 while" },
];

const senseBox = document.createElement("div");
senseBox.id = "sense";
senseBox.hidden = true;
document.body.appendChild(senseBox);
const senseGhost = document.createElement("div");
senseGhost.id = "senseGhost";
senseGhost.hidden = true;
document.body.appendChild(senseGhost);
const caretMirror = document.createElement("div");
caretMirror.setAttribute("aria-hidden", "true");
caretMirror.style.cssText = "position:absolute;visibility:hidden;left:0;top:0;white-space:pre;overflow:hidden;";
document.body.appendChild(caretMirror);

let senseItems = [];
let senseIndex = 0;
let senseTokenStart = 0;
let senseOpen = false;
let senseForce = false;
let senseLabel = "";
let composing = false;

function caretClient() {
  const style = getComputedStyle(code);
  caretMirror.style.font = style.font;
  caretMirror.style.lineHeight = style.lineHeight;
  caretMirror.style.letterSpacing = style.letterSpacing;
  caretMirror.style.tabSize = style.tabSize;
  caretMirror.style.padding = style.padding;
  caretMirror.style.width = `${code.clientWidth}px`;
  caretMirror.textContent = code.value.slice(0, code.selectionStart);
  const mark = document.createElement("span");
  mark.textContent = ".";
  caretMirror.appendChild(mark);
  const rect = code.getBoundingClientRect();
  const point = {
    x: rect.left + mark.offsetLeft - code.scrollLeft,
    y: rect.top + mark.offsetTop - code.scrollTop,
    line: parseFloat(style.lineHeight) || LINE,
  };
  caretMirror.removeChild(mark);
  senseGhost.style.font = style.font;
  senseGhost.style.lineHeight = style.lineHeight;
  senseGhost.style.tabSize = style.tabSize;
  return point;
}

function quoteOpen(text) {
  let quote = "";
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote) {
      if (ch === quote) quote = "";
    } else if (ch === "'" || ch === '"') {
      quote = ch;
    }
  }
  return quote;
}

function unclosedParen(text) {
  let quote = "";
  let depth = 0;
  let openAt = -1;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quote) {
      if (ch === quote) quote = "";
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      continue;
    }
    if (ch === "(" || ch === "[") {
      if (depth === 0 && ch === "(") openAt = i;
      depth += 1;
    } else if (ch === ")" || ch === "]") {
      depth -= 1;
      if (depth <= 0) {
        depth = 0;
        openAt = -1;
      }
    }
  }
  if (openAt < 0) return null;
  return { before: text.slice(0, openAt), inside: text.slice(openAt + 1) };
}

function commaCount(text) {
  let quote = "";
  let depth = 0;
  let count = 0;
  for (const ch of text) {
    if (quote) {
      if (ch === quote) quote = "";
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      continue;
    }
    if (ch === "(" || ch === "[") depth += 1;
    else if (ch === ")" || ch === "]") depth = Math.max(0, depth - 1);
    else if (ch === "," && depth === 0) count += 1;
  }
  return count;
}

function splitCallArgs(example) {
  const open = example.indexOf("(");
  const close = example.lastIndexOf(")");
  if (open < 0 || close <= open) return [];
  const body = example.slice(open + 1, close);
  const args = [];
  let quote = "";
  let depth = 0;
  let current = "";
  for (const ch of body) {
    if (quote) {
      current += ch;
      if (ch === quote) quote = "";
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      current += ch;
      continue;
    }
    if (ch === "(" || ch === "[") depth += 1;
    else if (ch === ")" || ch === "]") depth = Math.max(0, depth - 1);
    if (ch === "," && depth === 0 && !quote) {
      args.push(current.trim());
      current = "";
      continue;
    }
    current += ch;
  }
  if (current.trim()) args.push(current.trim());
  return args;
}

function logicalChunk() {
  const pos = code.selectionStart;
  if (pos !== code.selectionEnd) return null;
  const value = code.value;
  let lineStart = value.lastIndexOf("\n", pos - 1) + 1;
  while (lineStart > 0) {
    const prevEnd = lineStart - 1;
    const prevStart = value.lastIndexOf("\n", prevEnd - 1) + 1;
    if (!value.slice(prevStart, prevEnd).trimEnd().endsWith("\\")) break;
    lineStart = prevStart;
  }
  const chunk = value.slice(lineStart, pos);
  const trimmed = chunk.trimStart();
  if (trimmed.startsWith("*") || trimmed.startsWith("//")) return null;
  if (quoteOpen(chunk)) return null;
  const tokenMatch = chunk.match(/[A-Za-z_]\w*$/);
  const token = tokenMatch ? tokenMatch[0] : "";
  const head = chunk.slice(0, chunk.length - token.length);
  const paren = unclosedParen(head);
  const after = value.slice(pos);
  return {
    token,
    tokenStart: pos - token.length,
    head,
    paren,
    parenFollows: /^\s*\(/.test(after),
  };
}

function logicalLines(source) {
  const lines = [];
  let pending = "";
  for (const line of source.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.endsWith("\\")) {
      pending += `${trimmed.slice(0, -1)} `;
      continue;
    }
    lines.push((pending + line).trim());
    pending = "";
  }
  if (pending) lines.push(pending.trim());
  return lines;
}

function classifiedNames() {
  const iconic = new Set((server.iconic || []).map((item) => item.name));
  const control = new Set((server.control || []).map((item) => item.name));
  for (const raw of logicalLines(code.value)) {
    if (!raw || raw.startsWith("*") || raw.startsWith("//")) continue;
    const assign = raw.match(/^([A-Za-z_]\w*)\s*:=/);
    if (assign) {
      control.add(assign[1]);
      continue;
    }
    const call = raw.match(/^([A-Za-z_]\w*)\s*\((.*)\)\s*$/);
    if (!call) continue;
    const op = operators.find((item) => item.name.toLowerCase() === call[1].toLowerCase());
    if (!op) continue;
    splitCallArgs(`${call[1]} (${call[2]})`).forEach((arg, index) => {
      if (!/^[A-Za-z_]\w*$/.test(arg)) return;
      const param = op.params[index];
      if (!param) return;
      if (param.role.includes("圖像")) iconic.add(arg);
      else control.add(arg);
    });
  }
  return { iconic, control };
}

function variableDetail(name) {
  const iconic = (server.iconic || []).find((item) => item.name === name);
  if (iconic) return `${iconic.kind} · ${iconic.info}`;
  const control = (server.control || []).find((item) => item.name === name);
  if (control) return control.text;
  return "程式裡的變數";
}

function senseContextItems(ctx) {
  const token = ctx.token;
  const lower = token.toLowerCase();
  if (ctx.paren) {
    const nameMatch = ctx.paren.before.match(/(?:^|\s)([A-Za-z_]\w*)\s*$/);
    const opName = nameMatch ? nameMatch[1] : "";
    const op = operators.find((item) => item.name.toLowerCase() === opName.toLowerCase());
    const argIndex = commaCount(ctx.paren.inside);
    const param = op && op.params ? op.params[argIndex] : null;
    const exampleArg = op ? splitCallArgs(op.example)[argIndex] : "";
    const known = classifiedNames();
    const imageRole = param && param.role.includes("圖像");
    const controlRole = param && param.role.includes("控制");
    const picked = [];
    const used = new Set();
    const add = (kind, label, detail) => {
      if (!label || used.has(label)) return;
      if (lower && !label.toLowerCase().startsWith(lower)) return;
      used.add(label);
      picked.push({ kind, label, detail, insert: label });
    };
    if (exampleArg) add("範本", exampleArg, param ? param.desc : "這個參數的寫法");
    const pool = imageRole ? known.iconic : controlRole ? known.control : new Set([...known.iconic, ...known.control]);
    for (const name of pool) {
      const sessionText = variableDetail(name);
      const detail = sessionText === "程式裡的變數" ? (known.iconic.has(name) && !known.control.has(name) ? "圖像變數" : "控制變數") : sessionText;
      add("變數", name, detail);
    }
    const parts = (op && op.params ? op.params : []).map((item, index) => (
      index === argIndex ? `<b>${esc(item.name)}</b>` : esc(item.name)
    ));
    const sig = op
      ? `${esc(op.name)} ( ${parts.join(", ")} )${param ? `　${esc(param.desc)}` : ""}`
      : "";
    return { items: picked.slice(0, 40), sig };
  }
  const statement = ctx.head.replace(/\\\s*\n/g, "").trim() === "";
  if (!statement) return { items: [], sig: "" };
  if (!token && !senseForce) return { items: [], sig: "" };
  const ops = operators
    .filter((op) => op.name.toLowerCase().startsWith(lower))
    .filter((op) => !(ctx.parenFollows && op.name.toLowerCase() === lower))
    .sort((a, b) => a.name.length - b.name.length || a.order - b.order)
    .map((op) => ({
      kind: "算子",
      label: op.name,
      detail: op.short,
      insert: ctx.parenFollows ? op.name : op.example,
    }));
  const keys = SENSE_KEYS
    .filter((item) => item.name.startsWith(lower))
    .filter((item) => !(ctx.parenFollows && item.name === lower))
    .sort((a, b) => a.name.length - b.name.length)
    .map((item) => ({
      kind: "語法",
      label: item.name,
      detail: item.detail,
      insert: item.insert,
      caret: item.caret,
    }));
  return { items: [...ops, ...keys].sort((a, b) => a.label.length - b.label.length), sig: "" };
}

function closeSense() {
  senseOpen = false;
  senseItems = [];
  senseBox.hidden = true;
  senseBox.innerHTML = "";
  senseGhost.hidden = true;
  senseGhost.textContent = "";
}

function placeSense() {
  if (senseBox.hidden && senseGhost.hidden) return;
  const caret = caretClient();
  if (!senseGhost.hidden) {
    senseGhost.style.left = `${caret.x}px`;
    senseGhost.style.top = `${caret.y}px`;
  }
  if (senseBox.hidden) return;
  const margin = 8;
  let left = caret.x;
  let top = caret.y + caret.line;
  const width = senseBox.offsetWidth;
  const height = senseBox.offsetHeight;
  if (left + width > window.innerWidth - margin) left = Math.max(margin, window.innerWidth - width - margin);
  if (top + height > window.innerHeight - margin) top = Math.max(margin, caret.y - height);
  senseBox.style.left = `${left}px`;
  senseBox.style.top = `${top}px`;
}

function renderSense() {
  if (!senseOpen || !senseItems.length) {
    closeSense();
    return;
  }
  const item = senseItems[senseIndex];
  const ghost = item && senseItems.token && item.insert.toLowerCase().startsWith(senseItems.token.toLowerCase())
    ? item.insert.slice(senseItems.token.length)
    : "";
  senseBox.hidden = false;
  const rows = senseItems.map((entry, index) => {
    const extra = entry.insert !== entry.label ? `<span class="sense-insert">${esc(entry.insert)}</span>` : "";
    return `<button class="sense-item${index === senseIndex ? " on" : ""}" type="button" data-index="${index}"><span class="sense-kind">${esc(entry.kind)}</span><span class="sense-main"><span class="sense-label">${esc(entry.label)}</span><span class="sense-detail">${esc(entry.detail || "")}</span>${extra}</span></button>`;
  }).join("");
  senseBox.innerHTML = `${senseItems.sig ? `<div id="senseSig">${senseItems.sig}</div>` : ""}${rows}<div id="senseFoot">Tab 帶入後面的寫法　↑↓ 選擇　Esc 關閉</div>`;
  const current = senseBox.querySelector(".sense-item.on");
  if (current) {
    if (current.offsetTop < senseBox.scrollTop) senseBox.scrollTop = current.offsetTop;
    else if (current.offsetTop + current.offsetHeight > senseBox.scrollTop + senseBox.clientHeight) {
      senseBox.scrollTop = current.offsetTop + current.offsetHeight - senseBox.clientHeight;
    }
  }
  senseGhost.hidden = !ghost;
  senseGhost.textContent = ghost;
  placeSense();
}

function updateSense() {
  if (composing || document.activeElement !== code) {
    closeSense();
    return;
  }
  const ctx = logicalChunk();
  if (!ctx) {
    closeSense();
    return;
  }
  const built = senseContextItems(ctx);
  senseForce = false;
  if (!built.items.length) {
    closeSense();
    return;
  }
  const keep = built.items.findIndex((item) => item.label === senseLabel);
  senseItems = built.items;
  senseItems.sig = built.sig;
  senseItems.token = ctx.token;
  senseIndex = keep >= 0 ? keep : 0;
  senseLabel = senseItems[senseIndex].label;
  senseTokenStart = ctx.tokenStart;
  senseOpen = true;
  renderSense();
}

function moveSense(step) {
  if (!senseItems.length) return;
  senseIndex = (senseIndex + step + senseItems.length) % senseItems.length;
  senseLabel = senseItems[senseIndex].label;
  renderSense();
}

function acceptSense() {
  const item = senseItems[senseIndex];
  if (!item) return;
  const end = code.selectionStart;
  const start = senseTokenStart;
  code.value = code.value.slice(0, start) + item.insert + code.value.slice(end);
  const caret = start + (item.caret == null ? item.insert.length : item.caret);
  code.selectionStart = code.selectionEnd = caret;
  senseLabel = "";
  renderEditor();
  code.focus();
  updateSense();
}

senseBox.addEventListener("mousedown", (event) => event.preventDefault());
senseBox.addEventListener("click", (event) => {
  const node = event.target.closest(".sense-item");
  if (!node) return;
  senseIndex = Number(node.dataset.index);
  senseLabel = senseItems[senseIndex] ? senseItems[senseIndex].label : "";
  acceptSense();
});

function loadImg(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("影像解碼失敗"));
    image.src = src;
  });
}

function clearGraphics() {
  disp = null;
  baseImage = null;
  gfxEmpty.hidden = false;
  gfxTitle.textContent = "";
  stZoom.textContent = "";
  draw();
}

async function loadDisplay(display) {
  const token = ++loadToken;
  const jobs = [];
  let nextBase = null;
  if (display.image) jobs.push(loadImg(display.image).then((image) => { nextBase = image; }));
  const shapes = (display.shapes || []).map((shape) => ({ ...shape }));
  for (const shape of shapes) {
    if (shape.type === "fill" && shape.image) {
      jobs.push(loadImg(shape.image).then((image) => { shape._img = image; }));
    }
  }
  await Promise.all(jobs);
  if (token !== loadToken) return;
  disp = display;
  disp.shapes = shapes;
  baseImage = nextBase;
  if (baseImage) {
    probe.width = baseImage.naturalWidth;
    probe.height = baseImage.naturalHeight;
    probeCtx.drawImage(baseImage, 0, 0);
  }
  gfxEmpty.hidden = true;
  gfxTitle.textContent = `${display.width} × ${display.height}`;
  const sizeChanged = cam.fitW !== display.width || cam.fitH !== display.height;
  cam.fitW = display.width;
  cam.fitH = display.height;
  if (!cam.user || sizeChanged) {
    cam.user = false;
    fit(false);
  } else {
    draw();
  }
}

function fit(markUser = false) {
  const rect = canvas.getBoundingClientRect();
  if (!disp || !disp.width || rect.width < 2) return;
  const pad = 18;
  cam.scale = Math.max(0.02, Math.min((rect.width - pad * 2) / disp.width, (rect.height - pad * 2) / disp.height));
  cam.x = (rect.width - disp.width * cam.scale) / 2;
  cam.y = (rect.height - disp.height * cam.scale) / 2;
  cam.user = markUser;
  draw();
}

function zoomBy(factor) {
  if (!disp) return;
  const rect = canvas.getBoundingClientRect();
  const mx = rect.width / 2;
  const my = rect.height / 2;
  const next = Math.min(48, Math.max(0.05, cam.scale * factor));
  const k = next / cam.scale;
  cam.x = mx - (mx - cam.x) * k;
  cam.y = my - (my - cam.y) * k;
  cam.scale = next;
  cam.user = true;
  draw();
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  dprScale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.round(rect.width * dprScale));
  canvas.height = Math.max(1, Math.round(rect.height * dprScale));
  draw();
}

function draw() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  ctx.setTransform(dprScale, 0, 0, dprScale, 0, 0);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#101216";
  ctx.fillRect(0, 0, width, height);
  if (!disp) {
    stZoom.textContent = "";
    return;
  }
  ctx.save();
  ctx.translate(cam.x, cam.y);
  ctx.scale(cam.scale, cam.scale);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, disp.width, disp.height);
  if (baseImage) {
    ctx.imageSmoothingEnabled = cam.scale < 3;
    ctx.drawImage(baseImage, 0, 0);
  }
  if (previewImg) {
    ctx.drawImage(previewImg, 0, 0);
    ctx.restore();
    stZoom.textContent = `${Math.round(cam.scale * 100)}%`;
    return;
  }
  for (const shape of disp.shapes || []) {
    if (shape.type === "fill" && shape._img) ctx.drawImage(shape._img, 0, 0);
  }
  for (const shape of disp.shapes || []) {
    if (shape.type === "polyline" && shape.points?.length) {
      ctx.beginPath();
      ctx.moveTo(shape.points[0][0], shape.points[0][1]);
      for (let i = 1; i < shape.points.length; i += 1) ctx.lineTo(shape.points[i][0], shape.points[i][1]);
      if (shape.closed) ctx.closePath();
      ctx.strokeStyle = shape.color || "#00e05a";
      ctx.lineWidth = (shape.width || 2) / cam.scale;
      ctx.stroke();
    } else if (shape.type === "text") {
      const size = 14 / cam.scale;
      ctx.font = `${size}px "Segoe UI", "Microsoft JhengHei", sans-serif`;
      ctx.lineWidth = 3 / cam.scale;
      ctx.strokeStyle = "rgba(0,0,0,0.75)";
      ctx.strokeText(shape.text, shape.col, shape.row);
      ctx.fillStyle = shape.color || "#fff";
      ctx.fillText(shape.text, shape.col, shape.row);
    }
  }
  if (featureLines.length) {
    ctx.strokeStyle = "#ffe14a";
    ctx.lineWidth = 1.5 / cam.scale;
    for (const line of featureLines) {
      if (!line || line.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(line[0][0], line[0][1]);
      for (let i = 1; i < line.length; i += 1) ctx.lineTo(line[i][0], line[i][1]);
      ctx.stroke();
    }
  }
  ctx.restore();
  if (mouse) {
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(mouse.x + 0.5, 0);
    ctx.lineTo(mouse.x + 0.5, height);
    ctx.moveTo(0, mouse.y + 0.5);
    ctx.lineTo(width, mouse.y + 0.5);
    ctx.stroke();
  }
  stZoom.textContent = `${Math.round(cam.scale * 100)}%`;
}

function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  return {
    x,
    y,
    col: (x - cam.x) / cam.scale,
    row: (y - cam.y) / cam.scale,
  };
}

function updateProbe(event) {
  if (!disp) {
    stPix.textContent = "Row —   Column —";
    mouse = null;
    return;
  }
  const point = imagePoint(event);
  mouse = point;
  const col = Math.floor(point.col);
  const row = Math.floor(point.row);
  if (col < 0 || row < 0 || col >= disp.width || row >= disp.height || !baseImage) {
    stPix.textContent = `Row ${point.row.toFixed(1)}   Column ${point.col.toFixed(1)}`;
    draw();
    return;
  }
  const pixel = probeCtx.getImageData(col, row, 1, 1).data;
  const gray = Math.round(0.299 * pixel[0] + 0.587 * pixel[1] + 0.114 * pixel[2]);
  const color = pixel[0] === pixel[1] && pixel[1] === pixel[2]
    ? `Gray ${pixel[0]}`
    : `R ${pixel[0]}  G ${pixel[1]}  B ${pixel[2]}  Gray ${gray}`;
  stPix.textContent = `Row ${row}   Column ${col}   ${color}`;
  draw();
}

async function uploadFile(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body });
  const data = await res.json();
  if (!res.ok) {
    showDetail(data);
    return;
  }
  insertAtCursor(`read_image (Image, '${data.path}')`);
}

window.addEventListener("visdev-layout", () => resizeCanvas());

code.addEventListener("input", () => {
  renderEditor();
  updateSense();
});
code.addEventListener("scroll", () => {
  hl.scrollTop = code.scrollTop;
  hl.scrollLeft = code.scrollLeft;
  gutter.scrollTop = code.scrollTop;
  placeBars();
  placeSense();
});
code.addEventListener("keydown", (event) => {
  if (event.key === " " && (event.ctrlKey || event.metaKey) && !event.altKey) {
    event.preventDefault();
    senseForce = true;
    updateSense();
    return;
  }
  if (composing || event.isComposing || !senseOpen) return;
  if (event.key === "ArrowDown") {
    event.preventDefault();
    moveSense(1);
  } else if (event.key === "ArrowUp") {
    event.preventDefault();
    moveSense(-1);
  } else if (event.key === "Tab") {
    event.preventDefault();
    event.stopPropagation();
    acceptSense();
  } else if (event.key === "Escape") {
    event.preventDefault();
    event.stopPropagation();
    senseLabel = "";
    closeSense();
  }
});
code.addEventListener("keyup", (event) => {
  if (["ArrowDown", "ArrowUp", "Tab", "Enter", "Escape"].includes(event.key)) return;
  if (event.key.startsWith("Arrow") || event.key === "Home" || event.key === "End" || event.key === "PageUp" || event.key === "PageDown") {
    updateSense();
  }
});
code.addEventListener("click", () => updateSense());
code.addEventListener("compositionstart", () => {
  composing = true;
  closeSense();
});
code.addEventListener("compositionend", () => {
  composing = false;
  updateSense();
});
code.addEventListener("blur", () => {
  setTimeout(() => {
    if (document.activeElement !== code) closeSense();
  }, 150);
});
let arrowDrag = null;

function lineFromPointer(clientY) {
  const rect = gutter.getBoundingClientRect();
  const y = clientY - rect.top + gutter.scrollTop - PAD;
  const count = Math.max(1, code.value.split("\n").length);
  return Math.max(1, Math.min(count, Math.floor(y / LINE) + 1));
}

function paintArrowAim(line) {
  gutter.querySelectorAll(".gl.aim").forEach((node) => node.classList.remove("aim"));
  const node = gutter.querySelector(`.gl[data-line="${line}"]`);
  if (node) node.classList.add("aim");
}

gutter.addEventListener("mousedown", (event) => {
  const arrow = event.target.closest(".pc");
  const lineNode = event.target.closest(".gl.cur");
  if (!arrow || !lineNode) return;
  event.preventDefault();
  arrowDrag = { from: Number(lineNode.dataset.line), hover: Number(lineNode.dataset.line) };
  document.body.classList.add("arrow-drag");
});
window.addEventListener("mousemove", (event) => {
  if (!arrowDrag) return;
  const line = lineFromPointer(event.clientY);
  arrowDrag.hover = line;
  paintArrowAim(line);
});
window.addEventListener("mouseup", () => {
  if (!arrowDrag) return;
  const target = arrowDrag.hover;
  const from = arrowDrag.from;
  arrowDrag = null;
  document.body.classList.remove("arrow-drag");
  gutter.querySelectorAll(".gl.aim").forEach((node) => node.classList.remove("aim"));
  if (target && target !== from) exec("goto", target);
});
gutter.addEventListener("click", (event) => {
  if (event.target.closest(".pc")) return;
  const lineNode = event.target.closest(".gl");
  if (!lineNode) return;
  const line = Number(lineNode.dataset.line);
  if (breakpoints.has(line)) breakpoints.delete(line);
  else breakpoints.add(line);
  renderGutter();
});
document.getElementById("errOk").onclick = () => closeErrorDialog();
document.getElementById("errDialog").addEventListener("mousedown", (event) => {
  if (event.target.id === "errDialog") closeErrorDialog();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !document.getElementById("errDialog").hidden) {
    event.preventDefault();
    closeErrorDialog();
  }
});

document.getElementById("btnRun").onclick = () => exec("run");
document.getElementById("btnStep").onclick = () => exec("step");
document.getElementById("btnReset").onclick = () => exec("reset");
document.getElementById("btnStop").onclick = async () => {
  if (sessionId) await fetch(`/api/stop/${sessionId}`, { method: "POST" });
};
document.getElementById("btnFit").onclick = () => { cam.user = false; fit(false); };
document.getElementById("btnOne").onclick = () => {
  if (!disp) return;
  const rect = canvas.getBoundingClientRect();
  cam.scale = 1;
  cam.x = (rect.width - disp.width) / 2;
  cam.y = (rect.height - disp.height) / 2;
  cam.user = true;
  draw();
};
document.getElementById("btnZoomIn").onclick = () => zoomBy(1.25);
document.getElementById("btnZoomOut").onclick = () => zoomBy(0.8);
document.getElementById("btnUpload").onclick = () => document.getElementById("fileImage").click();
document.getElementById("fileImage").onchange = () => {
  const file = document.getElementById("fileImage").files[0];
  if (file) uploadFile(file);
};
document.getElementById("btnOpen").onclick = () => document.getElementById("fileOpen").click();
document.getElementById("fileOpen").onchange = async () => {
  const file = document.getElementById("fileOpen").files[0];
  if (!file) return;
  code.value = (await file.text()).replace(/\r\n/g, "\n");
  document.getElementById("progName").value = file.name;
  sessionId = null;
  breakpoints = new Set();
  renderEditor();
  await exec("reset");
};
document.getElementById("btnSave").onclick = async () => {
  const name = document.getElementById("progName").value.trim() || "program.hdv";
  const res = await fetch("/api/programs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, source: code.value }),
  });
  const data = await res.json();
  if (!res.ok) {
    showDetail(data);
    return;
  }
  document.getElementById("progName").value = data.name;
  stState.textContent = `已儲存 ${data.path}`;
  stState.className = "ok";
  loadFiles();
};
document.getElementById("btnSamples").onclick = (event) => {
  event.stopPropagation();
  sampleMenu.hidden = !sampleMenu.hidden;
};
document.addEventListener("click", (event) => {
  if (!event.target.closest(".pop")) sampleMenu.hidden = true;
});

window.addEventListener("keydown", (event) => {
  if (event.key === "F5") { event.preventDefault(); exec("run"); }
  if (event.key === "F6") { event.preventDefault(); exec("step"); }
  if (event.key === "F7") { event.preventDefault(); exec("reset"); }
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
    event.preventDefault();
    document.getElementById("btnSave").click();
  }
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.onclick = () => showTab(tab.dataset.tab);
});
opSearch.addEventListener("input", renderOps);
opList.addEventListener("click", (event) => {
  const node = event.target.closest(".op");
  if (node) showOp(node.dataset.name);
});
opList.addEventListener("dblclick", (event) => {
  const node = event.target.closest(".op");
  if (!node) return;
  const op = operators.find((item) => item.name === node.dataset.name);
  if (op) insertAtCursor(op.example);
});

iconicList.addEventListener("click", async (event) => {
  const node = event.target.closest(".var");
  if (!node || !sessionId) return;
  selectedVar = node.dataset.name;
  renderVars();
  const res = await fetch(`/api/session/${sessionId}/object/${encodeURIComponent(selectedVar)}`);
  const data = await res.json();
  if (!res.ok) {
    showDetail(data);
    return;
  }
  gfxBanner.hidden = false;
  gfxBanner.textContent = `預覽 ${selectedVar} · 再執行會回到程式畫面`;
  await loadDisplay(data.display);
});
controlList.addEventListener("click", (event) => {
  const node = event.target.closest(".var");
  if (!node) return;
  selectedVar = node.dataset.name;
  const item = server.control.find((entry) => entry.name === selectedVar);
  varDetail.textContent = item ? `${item.name}  ${item.type}\n${item.full}` : "";
  renderVars();
});

canvas.addEventListener("wheel", (event) => {
  if (!disp) return;
  event.preventDefault();
  const point = imagePoint(event);
  const next = Math.min(48, Math.max(0.05, cam.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12)));
  const k = next / cam.scale;
  cam.x = point.x - (point.x - cam.x) * k;
  cam.y = point.y - (point.y - cam.y) * k;
  cam.scale = next;
  cam.user = true;
  draw();
}, { passive: false });

let dragging = null;
canvas.addEventListener("mousedown", (event) => {
  if (event.button !== 0) return;
  const point = imagePoint(event);
  dragging = { x: point.x, y: point.y, camX: cam.x, camY: cam.y, col: point.col, row: point.row };
  canvas.classList.add("dragging");
});
window.addEventListener("mouseup", (event) => {
  const start = dragging;
  dragging = null;
  canvas.classList.remove("dragging");
  if (!start || !disp) return;
  const point = imagePoint(event);
  if (Math.hypot(point.x - start.x, point.y - start.y) > 4) return;
  pickObject(start.col, start.row);
});
canvas.addEventListener("mousemove", (event) => {
  if (dragging) {
    const point = imagePoint(event);
    cam.x = dragging.camX + point.x - dragging.x;
    cam.y = dragging.camY + point.y - dragging.y;
    cam.user = true;
  }
  updateProbe(event);
});
canvas.addEventListener("mouseleave", () => {
  mouse = null;
  draw();
});
const stage = document.getElementById("gfxStage");
stage.addEventListener("dragover", (event) => event.preventDefault());
stage.addEventListener("drop", (event) => {
  event.preventDefault();
  const file = event.dataTransfer.files[0];
  if (file) uploadFile(file);
});

new ResizeObserver(() => resizeCanvas()).observe(stage);

let histo = null;
let thresholdChannel = null;

function regionColorValue() {
  const saved = localStorage.getItem("visdev.regionColor");
  return saved && /^#[0-9a-fA-F]{6}$/.test(saved) ? saved : "#ff3b30";
}

function renderHisto(data) {
  histo = data && data.bins ? data : null;
  publishThreshold();
}

function splitArgs(inner) {
  const parts = [];
  let current = "";
  let quote = "";
  for (const ch of inner) {
    if (quote) {
      current += ch;
      if (ch === quote) quote = "";
      continue;
    }
    if (ch === "'" || ch === '"') {
      quote = ch;
      current += ch;
      continue;
    }
    if (ch === ",") {
      parts.push(current);
      current = "";
      continue;
    }
    current += ch;
  }
  if (current.length || parts.length) parts.push(current);
  return parts;
}

function guessImageName(lines) {
  let name = "Image";
  for (const line of lines) {
    const match = line.match(/^\s*(read_image|rgb1_to_gray|gauss_image|median_image|mean_image|emphasize|scale_image|invert_image)\s*\((.*)\)\s*$/i);
    if (!match) continue;
    const args = splitArgs(match[2]);
    const picked = match[1].toLowerCase() === "read_image" ? args[0] : args[1];
    if (picked && picked.trim()) name = picked.trim();
  }
  return name;
}

function applyBinaryTool(tool) {
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const index = lines.findIndex((line) => /^\s*(threshold|opencv_threshold|binary_threshold)\s*\(/i.test(line));
  let image = guessImageName(lines);
  let region = "Region";
  let level = histo && Number.isFinite(histo.threshold) ? histo.threshold : (histo && Number.isFinite(histo.min) ? histo.min : 128);
  let maxGray = histo && histo.kind === "range" ? histo.max : 255;
  let used = "UsedThreshold";
  let indent = "";
  if (index >= 0) {
    const match = lines[index].match(/^(\s*)(?:threshold|opencv_threshold|binary_threshold)\s*\((.*)\)\s*$/i);
    if (match) {
      indent = match[1];
      const args = splitArgs(match[2]).map((item) => item.trim());
      const name = (lines[index].match(/opencv_threshold|binary_threshold|threshold/i) || ["threshold"])[0].toLowerCase();
      if (args[0]) image = args[0];
      if (args[1]) region = args[1];
      if (name === "threshold") {
        if (/^-?\d+(\.\d+)?$/.test(args[2] || "")) level = Math.round(Number(args[2]));
        if (/^-?\d+(\.\d+)?$/.test(args[3] || "")) maxGray = Math.round(Number(args[3]));
      } else if (name === "opencv_threshold" && /^-?\d+(\.\d+)?$/.test(args[2] || "")) {
        level = Math.round(Number(args[2]));
      } else if (name === "binary_threshold" && args[4]) {
        used = args[4];
      }
    }
  }
  let call = `binary_threshold (${image}, ${region}, 'max_separability', 'light', ${used})`;
  if (tool === "halcon") call = `threshold (${image}, ${region}, ${level}, ${maxGray})`;
  else if (tool === "opencv") call = `opencv_threshold (${image}, ${region}, ${level}, 'binary')`;
  else if (tool === "inv") call = `opencv_threshold (${image}, ${region}, ${level}, 'binary_inv')`;
  else if (tool === "dark") call = `binary_threshold (${image}, ${region}, 'max_separability', 'dark', ${used})`;
  else if (tool === "smooth") call = `binary_threshold (${image}, ${region}, 'smooth_histo', 'light', ${used})`;
  else if (tool === "triangle") call = `binary_threshold (${image}, ${region}, 'triangle', 'light', ${used})`;
  else if (tool === "local") call = `binary_threshold (${image}, ${region}, 'local_mean', 'light', ${used})`;
  else if (tool === "lgauss") call = `binary_threshold (${image}, ${region}, 'local_gauss', 'light', ${used})`;
  if (index >= 0) lines[index] = indent + call;
  else {
    const displayAt = lines.findIndex((line) => /^\s*dev_display\s*\(/i.test(line));
    lines.splice(displayAt >= 0 ? displayAt : lines.length, 0, call);
  }
  const opencv = tool === "opencv" || tool === "inv";
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    if (opencv) {
      const linked = line.match(/^(\s*)connection\s*\(\s*([^,]+)\s*,\s*([^,)]+)\s*(?:,\s*([^)]+))?\s*\)\s*$/i);
      if (linked && linked[2].trim() === region) {
        const dist = linked[4] ? linked[4].trim() : "1";
        lines[lineIndex] = `${linked[1]}opencv_connection (${linked[2].trim()}, ${linked[3].trim()}, 8, ${dist})`;
      }
    } else {
      const linked = line.match(/^(\s*)opencv_connection\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*(?:,\s*([^)]+))?\s*\)\s*$/i);
      if (linked && linked[2].trim() === region) {
        const dist = linked[5] ? linked[5].trim() : "1";
        lines[lineIndex] = `${linked[1]}connection (${linked[2].trim()}, ${linked[3].trim()}, ${dist})`;
      }
    }
  }
  code.value = lines.join("\n");
  renderEditor();
  exec("run");
}

function applyConnection(minArea, dist) {
  const area = Math.max(0, Math.round(Number(minArea) || 0));
  const distance = !Number.isFinite(Number(dist)) || Number(dist) < 1 ? 1 : Math.round(Number(dist) * 1000) / 1000;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  let region = "Region";
  let opencv = false;
  const bin = lines.findIndex((line) => /^\s*(threshold|opencv_threshold|binary_threshold)\s*\(/i.test(line));
  if (bin >= 0) {
    const inner = lines[bin].match(/\((.*)\)/);
    const args = inner ? splitArgs(inner[1]).map((item) => item.trim()) : [];
    if (args[1]) region = args[1];
    opencv = /^\s*opencv_threshold/i.test(lines[bin]);
  }
  const regionPattern = region.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const connPattern = new RegExp(`^\\s*(connection|opencv_connection)\\s*\\(\\s*${regionPattern}\\s*,`, "i");
  let connIndex = lines.findIndex((line) => connPattern.test(line));
  let output = "Connected";
  if (connIndex >= 0) {
    const match = lines[connIndex].match(/^(\s*)(connection|opencv_connection)\s*\((.*)\)\s*$/i);
    const args = splitArgs(match[3]).map((item) => item.trim());
    output = args[1] || output;
    if (match[2].toLowerCase() === "opencv_connection") {
      const linked = args[2] || "8";
      lines[connIndex] = `${match[1]}opencv_connection (${args[0]}, ${output}, ${linked}, ${distance})`;
    } else {
      lines[connIndex] = `${match[1]}connection (${args[0]}, ${output}, ${distance})`;
    }
  } else {
    const insertAt = bin >= 0 ? bin + 1 : lines.length;
    const call = opencv
      ? `opencv_connection (${region}, ${output}, 8, ${distance})`
      : `connection (${region}, ${output}, ${distance})`;
    lines.splice(insertAt, 0, call);
    connIndex = insertAt;
  }
  const outputPattern = output.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const shapePattern = new RegExp(`^\\s*select_shape\\s*\\(\\s*${outputPattern}\\s*,`, "i");
  const shapeIndex = lines.findIndex((line, index) => index > connIndex && shapePattern.test(line) && /area/i.test(line));
  if (shapeIndex >= 0) {
    const match = lines[shapeIndex].match(/^(\s*)select_shape\s*\((.*)\)\s*$/i);
    const args = splitArgs(match[2]);
    if (args.length >= 6) {
      args[4] = ` ${area}`;
      lines[shapeIndex] = `${match[1]}select_shape (${args.map((item) => item.trim()).join(", ")})`;
    }
  } else {
    lines.splice(connIndex + 1, 0, `select_shape (${output}, Selected, 'area', 'and', ${area}, 99999999)`);
  }
  code.value = lines.join("\n");
  renderEditor();
  exec("run");
}

let previewImg = null;
let previewToken = 0;

function rewriteSubpix(level) {
  if (!histo || !histo.literal || histo.kind !== "subpix" || !histo.line) return false;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const original = lines[histo.line - 1];
  if (!original) return false;
  const match = original.match(/^(\s*)threshold_sub_pix\s*\((.*)\)\s*$/i);
  if (!match) return false;
  const args = splitArgs(match[2]);
  if (args.length < 3 || !/^-?\d+(\.\d+)?$/.test(args[2].trim())) return false;
  args[2] = ` ${level}`;
  lines[histo.line - 1] = `${match[1]}threshold_sub_pix (${args.join(",")})`;
  code.value = lines.join("\n");
  renderEditor();
  return true;
}

function applySubpix(level) {
  const value = Math.max(0, Math.min(255, Math.round(Number(level) || 0)));
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const image = guessImageName(lines);
  let output = "Border";
  const index = lines.findIndex((line) => /^\s*threshold_sub_pix\s*\(/i.test(line));
  if (index >= 0) {
    const match = lines[index].match(/^(\s*)threshold_sub_pix\s*\((.*)\)\s*$/i);
    const args = splitArgs(match[2]).map((item) => item.trim());
    output = args[1] || output;
    lines[index] = `${match[1]}threshold_sub_pix (${args[0] || image}, ${output}, ${value})`;
  } else {
    lines.push(`threshold_sub_pix (${image}, ${output}, ${value})`);
    lines.push(`dev_display (${output})`);
  }
  code.value = lines.join("\n");
  renderEditor();
  exec("run");
}

async function openBinWindow() {
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  await exec("run");
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  const level = histo && Number.isFinite(Number(histo.threshold))
    ? histo.threshold
    : histo && Number.isFinite(Number(histo.min))
      ? histo.min
      : 128;
  const url = `${location.origin}/bin?id=${encodeURIComponent(sessionId || "")}&level=${encodeURIComponent(level)}`;
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_bin_window) {
      await window.pywebview.api.open_bin_window(url);
      return;
    }
  } catch (_err) {
    /* 沒有桌面視窗介面時改開瀏覽器視窗 */
  }
  const popup = window.open(url, "visdev-subpix", "popup=yes,width=480,height=720,left=80,top=60");
  if (!popup) applySubpix(level);
}

function rewriteOpenCV(level) {
  if (!histo || !histo.literal || histo.kind !== "opencv" || !histo.line) return false;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const original = lines[histo.line - 1];
  if (!original) return false;
  const match = original.match(/^(\s*)opencv_threshold\s*\((.*)\)\s*$/i);
  if (!match) return false;
  const args = splitArgs(match[2]);
  if (args.length < 3 || !/^-?\d+(\.\d+)?$/.test(args[2].trim())) return false;
  args[2] = ` ${level}`;
  lines[histo.line - 1] = `${match[1]}opencv_threshold (${args.join(",")})`;
  code.value = lines.join("\n");
  renderEditor();
  return true;
}

function rewriteThreshold(minGray, maxGray) {
  if (!histo || !histo.literal || histo.kind !== "range" || !histo.line) return false;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const index = histo.line - 1;
  const original = lines[index];
  if (!original) return false;
  const match = original.match(/^(\s*)threshold\s*\((.*)\)\s*$/i);
  if (!match) return false;
  const args = splitArgs(match[2]);
  if (args.length < 2) return false;
  if (!/^-?\d+(\.\d+)?$/.test(args.at(-1).trim()) || !/^-?\d+(\.\d+)?$/.test(args.at(-2).trim())) return false;
  args[args.length - 2] = ` ${minGray}`;
  args[args.length - 1] = ` ${maxGray}`;
  lines[index] = `${match[1]}threshold (${args.join(",")})`;
  code.value = lines.join("\n");
  renderEditor();
  return true;
}

async function applyRegionColor(hex) {
  localStorage.setItem("visdev.regionColor", hex);
  if (!sessionId) return;
  const res = await fetch("/api/region-color", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ id: sessionId, color: hex }),
  });
  const data = await res.json();
  if (res.ok && data.display) loadDisplay(data.display);
  publishThreshold();
}

function publishThreshold() {
  const minMatch = code.value.match(/select_shape\s*\([^)]*'area'[^)]*,\s*([0-9.]+)\s*,/i);
  const opencv = code.value.match(/opencv_connection\s*\(\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*([0-9.]+)\s*\)/i);
  const halcon = code.value.match(/(?<!opencv_)connection\s*\(\s*[^,]+,\s*[^,)]+\s*,\s*([0-9.]+)\s*\)/i);
  const payload = {
    type: "state",
    id: sessionId,
    histogram: histo,
    color: regionColorValue(),
    connMin: minMatch ? Number(minMatch[1]) : null,
    connDist: (opencv || halcon) ? Number((opencv || halcon)[1]) : null,
    at: Date.now(),
  };
  localStorage.setItem("visdev.threshold", JSON.stringify(payload));
  if (thresholdChannel) thresholdChannel.postMessage(payload);
  const distInput = document.getElementById("connDist");
  const minInput = document.getElementById("connMin");
  if (distInput && document.activeElement !== distInput && payload.connDist != null) distInput.value = payload.connDist;
  if (minInput && document.activeElement !== minInput && payload.connMin != null) minInput.value = payload.connMin;
}

function commitThreshold(data) {
  if (!histo || !data) return;
  histo.min = data.min;
  histo.max = data.max;
  histo.threshold = data.threshold;
  if (data.mode === "level" && histo.kind === "subpix" && rewriteSubpix(data.threshold)) exec("run");
  else if (data.mode === "level" && rewriteOpenCV(data.threshold)) exec("run");
  else if (data.mode !== "level" && rewriteThreshold(data.min, data.max)) exec("run");
}

let lastThresholdCmd = 0;
function handleThresholdCmd(data) {
  if (!data || data.at === lastThresholdCmd) return;
  lastThresholdCmd = data.at || 0;
  if (data.type === "tool") applyBinaryTool(data.name);
  else if (data.type === "commit") commitThreshold(data);
  else if (data.type === "color") applyRegionColor(data.color);
  else if (data.type === "connection") applyConnection(data.min, data.dist);
  else if (data.type === "open-subpix") openBinWindow();
  else if (data.type === "open-features") openFeatureWindow();
  else if (data.type === "hello") publishThreshold();
}

async function openThresholdWindow() {
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  if (!sessionId || !histo) await exec("run");
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  publishThreshold();
  const url = `${location.origin}/threshold?id=${encodeURIComponent(sessionId || "")}`;
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_threshold_window) {
      await window.pywebview.api.open_threshold_window(url);
      return;
    }
  } catch (_err) {
    /* 沒有桌面視窗介面時改開瀏覽器視窗 */
  }
  window.open(url, "visdev-threshold", "popup=yes,width=920,height=720,left=80,top=40");
}

const binChannel = new BroadcastChannel("visdev-bin");
binChannel.onmessage = (event) => {
  if (event.data && event.data.type === "apply") applySubpix(event.data.threshold);
};
thresholdChannel = new BroadcastChannel("visdev-threshold");
thresholdChannel.onmessage = (event) => handleThresholdCmd(event.data);
window.addEventListener("storage", (event) => {
  if (event.key === "visdev.subpixApply" && event.newValue) {
    try {
      const data = JSON.parse(event.newValue);
      if (data && data.type === "apply") applySubpix(data.threshold);
    } catch (_err) {
      /* 忽略壞掉的暫存 */
    }
  }
  if (event.key === "visdev.thresholdCmd" && event.newValue) {
    try {
      handleThresholdCmd(JSON.parse(event.newValue));
    } catch (_err) {
      /* 忽略壞掉的暫存 */
    }
  }
});
document.getElementById("btnThreshold").addEventListener("click", () => openThresholdWindow());
document.getElementById("btnFeatures").addEventListener("click", () => openFeatureWindow());
document.getElementById("btnConnection").addEventListener("click", () => {
  applyConnection(document.getElementById("connMin").value, document.getElementById("connDist").value);
});
setInterval(() => {
  const raw = localStorage.getItem("visdev.thresholdCmd");
  if (!raw) return;
  try { handleThresholdCmd(JSON.parse(raw)); } catch (_err) { /* 忽略壞掉的暫存 */ }
}, 300);

async function loadFiles() {
  const files = await fetch("/api/files").then((res) => res.json());
  const block = (title, items) => {
    if (!items.length) return "";
    return `<div class="grouplab">${esc(title)}</div>` + items.map((item) =>
      `<button type="button" data-kind="${esc(item.kind)}" data-name="${esc(item.name)}">${esc(item.name)}</button>`
    ).join("");
  };
  sampleMenu.innerHTML = block("範例", files.samples || []) + block("已儲存", files.workspace || []);
}

sampleMenu.addEventListener("click", async (event) => {
  const node = event.target.closest("button");
  if (!node) return;
  const res = await fetch(`/api/files/${node.dataset.kind}/${encodeURIComponent(node.dataset.name)}`);
  const data = await res.json();
  if (!res.ok) {
    showDetail(data);
    return;
  }
  code.value = data.source.replace(/\r\n/g, "\n");
  document.getElementById("progName").value = data.name;
  sessionId = null;
  breakpoints = new Set();
  sampleMenu.hidden = true;
  renderEditor();
  await exec("reset");
});

const DEFAULT_FEATURES = [
  "region:area", "region:row", "region:column", "region:circularity", "region:contlength",
  "gray:gray_mean", "gray:gray_deviation",
  "xld:contlength", "subpixel",
];

function featureKeys() {
  try {
    const saved = JSON.parse(localStorage.getItem("visdev.features") || "null");
    if (Array.isArray(saved)) return saved;
  } catch (_err) {
    /* 用預設勾選 */
  }
  return DEFAULT_FEATURES;
}

function selectedFeatures() {
  const region = [];
  const xld = [];
  const gray = [];
  let subpixel = false;
  featureKeys().forEach((key) => {
    if (key === "subpixel") {
      subpixel = true;
      return;
    }
    const split = key.indexOf(":");
    if (split < 0) return;
    const kind = key.slice(0, split);
    const name = key.slice(split + 1);
    if (kind === "region") region.push(name);
    if (kind === "xld") xld.push(name);
    if (kind === "gray") gray.push(name);
  });
  return { region, xld, gray, subpixel };
}

async function openFeatureWindow() {
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  if (!sessionId) await exec("run");
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  const url = `${location.origin}/features?id=${encodeURIComponent(sessionId || "")}`;
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_feature_window) {
      await window.pywebview.api.open_feature_window(url);
      return;
    }
  } catch (_err) {
    /* 沒有桌面視窗介面時改開瀏覽器視窗 */
  }
  window.open(url, "visdev-features", "popup=yes,width=460,height=720,left=560,top=60");
}

function publishFeaturePick() {
  if (!pickedObject) return;
  const payload = {
    type: "pick",
    id: sessionId,
    name: pickedObject.name,
    index: pickedObject.index,
    kind: pickedObject.kind,
    at: Date.now(),
  };
  localStorage.setItem("visdev.featurePick", JSON.stringify(payload));
  featureChannel.postMessage(payload);
}

function pointInPoly(x, y, points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
    const xi = points[i][0];
    const yi = points[i][1];
    const xj = points[j][0];
    const yj = points[j][1];
    const cross = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi);
    if (cross) inside = !inside;
  }
  return inside;
}

function lineDistance(x, y, points) {
  let best = Infinity;
  for (let i = 1; i < points.length; i += 1) {
    const ax = points[i - 1][0];
    const ay = points[i - 1][1];
    const bx = points[i][0];
    const by = points[i][1];
    const dx = bx - ax;
    const dy = by - ay;
    const len2 = dx * dx + dy * dy || 1;
    let t = ((x - ax) * dx + (y - ay) * dy) / len2;
    t = Math.max(0, Math.min(1, t));
    best = Math.min(best, Math.hypot(x - (ax + t * dx), y - (ay + t * dy)));
  }
  return best;
}

function pickObject(col, row) {
  const shapes = disp?.shapes || [];
  let regionHit = null;
  let xldHit = null;
  let xldDist = Infinity;
  const limit = 6 / cam.scale;
  for (let i = shapes.length - 1; i >= 0; i -= 1) {
    const shape = shapes[i];
    if (shape.type !== "polyline" || shape.object == null || !shape.source) continue;
    if (shape.kind === "xld") {
      const dist = lineDistance(col, row, shape.points);
      if (dist <= limit && dist < xldDist) {
        xldDist = dist;
        xldHit = shape;
      }
    } else if (!regionHit && pointInPoly(col, row, shape.points)) {
      regionHit = shape;
    }
  }
  const shape = regionHit || xldHit;
  if (!shape) return;
  pickedObject = { name: shape.source, index: shape.object, kind: shape.kind || "region" };
  publishFeaturePick();
  loadFeatures();
}

async function loadFeatures() {
  const choice = selectedFeatures();
  if (!pickedObject || !sessionId || pickedObject.kind !== "region" || !choice.subpixel) {
    featureLines = [];
    draw();
    return;
  }
  const res = await fetch("/api/features", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      id: sessionId,
      name: pickedObject.name,
      index: pickedObject.index,
      kind: "region",
      region: [],
      xld: [],
      gray: [],
      subpixel: true,
    }),
  });
  const data = await res.json();
  featureLines = res.ok ? (data.contour || []) : [];
  draw();
}

const featureChannel = new BroadcastChannel("visdev-features");
featureChannel.onmessage = (event) => {
  if (event.data && event.data.type === "features" && pickedObject) loadFeatures();
};
window.addEventListener("storage", (event) => {
  if (event.key === "visdev.features" && pickedObject) loadFeatures();
});

async function boot() {
  operators = await fetch("/api/operators").then((res) => res.json());
  operatorNames = new Set(operators.map((op) => op.name.toLowerCase()));
  renderOps();
  if (operators[0]) showOp(operators[0].name);
  await loadFiles();
  const sample = await fetch("/api/files/samples/blob_inspect.hdv").then((res) => res.json());
  code.value = sample.source;
  document.getElementById("progName").value = sample.name;
  renderEditor();
  resizeCanvas();
  await exec("reset");
}

boot();
