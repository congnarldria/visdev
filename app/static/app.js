const code = document.getElementById("code");
const codeHistory = window.VisdevEdit.createEditorHistory();
let applyingHistory = false;

function commitProgram(mode) {
  if (applyingHistory) return;
  codeHistory.commit(code.value, code.selectionStart, code.selectionEnd, mode || "checkpoint");
}

function resetProgramHistory() {
  codeHistory.reset(code.value, code.selectionStart, code.selectionEnd);
}

function applyHistoryEntry(entry) {
  applyingHistory = true;
  code.value = entry.value;
  code.selectionStart = entry.start;
  code.selectionEnd = entry.end;
  applyingHistory = false;
  renderEditor();
  closeSense();
  refreshDirty();
}
const hl = document.getElementById("hl");
const gutter = document.getElementById("gutter");
const lineTimes = document.getElementById("lineTimes");
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
let runningSource = null;
let savedText = "";
let saveTarget = { kind: "sample", name: "blob_inspect.hdv" };

function refreshDirty() {
  const dirty = code.value !== savedText;
  const button = document.getElementById("btnSave");
  if (button) button.textContent = dirty ? "儲存 *" : "儲存";
  const name = document.getElementById("progName").value.trim() || "程式";
  document.title = `${dirty ? "* " : ""}${name} — VisDev`;
}

function markSaved() {
  savedText = code.value;
  refreshDirty();
}
let server = { line: null, done: false, error: null, errorLine: null, iconic: [], control: [], display: null, log: [], paused: null, lineTimes: [] };
let diagnostics = [];
let logHidden = false;
const problemFilter = { error: true, warning: true, message: true };
const CALL_KEYWORDS = new Set(["if", "else", "endif", "for", "endfor", "while", "endwhile", "try", "catch", "endtry", "procedure", "endprocedure", "endproc"]);
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

async function pushBreakpoints() {
  if (!sessionId || !server.running || runningSource == null) return;
  if (!VisdevRun.sameProgram(runningSource, code.value)) return;
  const lines = [...breakpoints];
  try {
    await fetch(`/api/session/${encodeURIComponent(sessionId)}/breakpoints`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ lines }),
    });
  } catch (_err) {
    /* 下一輪執行仍會帶上編輯器裡的中斷點 */
  }
}

function programArrow() {
  if (!server.line) return null;
  if (runningSource == null) return server.line;
  return VisdevRun.arrowLine(runningSource, code.value, server.line);
}

function noteRunningProgram() {
  if (runningSource == null) return;
  const note = VisdevRun.statusText(runningSource, code.value, server.line, !!server.running, server.paused);
  if (!note) {
    if (!server.running) {
      runningSource = null;
      setStatus();
    }
    return;
  }
  if (server.running) stState.textContent = "執行中";
  stLine.textContent = note;
}

function renderGutter() {
  const count = Math.max(1, code.value.split("\n").length);
  const arrow = programArrow();
  let html = "";
  for (let line = 1; line <= count; line += 1) {
    const cls = ["gl"];
    if (breakpoints.has(line)) cls.push("bp");
    if (arrow === line) cls.push("cur");
    if (server.errorLine === line || diagnostics.some((item) => item.line === line && item.sev === "error")) cls.push("err");
    else if (diagnostics.some((item) => item.line === line && item.sev === "warning")) cls.push("warn");
    html += `<div class="${cls.join(" ")}" data-line="${line}"><span class="pc"></span><span class="num">${line}</span></div>`;
  }
  gutter.innerHTML = html;
}

function placeBars() {
  const topFor = (line) => PAD + (line - 1) * LINE - code.scrollTop;
  const arrow = programArrow();
  if (arrow) {
    currentBar.style.display = "block";
    currentBar.style.top = `${topFor(arrow)}px`;
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

function formatLineTime(item) {
  const ms = Number(item.ms) || 0;
  let text;
  if (ms < 0.001) text = "<0.001 ms";
  else if (ms < 1) text = `${ms.toFixed(3)} ms`;
  else if (ms < 100) text = `${ms.toFixed(2)} ms`;
  else if (ms < 1000) text = `${ms.toFixed(1)} ms`;
  else text = `${(ms / 1000).toFixed(2)} s`;
  if (item.count > 1) text += ` ×${item.count}`;
  return text;
}

function renderTimes() {
  const count = Math.max(1, code.value.split("\n").length);
  const times = new Map((server.lineTimes || []).map((item) => [item.line, item]));
  const arrow = programArrow();
  let html = "";
  for (let line = 1; line <= count; line += 1) {
    const item = times.get(line);
    html += `<div class="${arrow === line ? "on" : ""}">${item ? formatLineTime(item) : ""}</div>`;
  }
  lineTimes.innerHTML = html;
}

function revealLine(line) {
  if (!line) return;
  const top = PAD + (line - 1) * LINE;
  const bottom = top + LINE;
  const viewTop = code.scrollTop;
  const viewBottom = viewTop + code.clientHeight;
  if (top >= viewTop && bottom <= viewBottom) return;
  code.scrollTop = Math.max(0, top - Math.max(LINE, (code.clientHeight - LINE) / 2));
  hl.scrollTop = code.scrollTop;
  hl.scrollLeft = code.scrollLeft;
  gutter.scrollTop = code.scrollTop;
  lineTimes.scrollTop = code.scrollTop;
  placeBars();
}

function renderEditor() {
  renderProblems();
  renderHighlight();
  renderGutter();
  renderTimes();
  placeBars();
}

function renderVars() {
  iconicCount.textContent = server.iconic.length ? String(server.iconic.length) : "";
  iconicList.innerHTML = server.iconic.map((item) => `
    <button class="var ${selectedVar === item.name ? "on" : ""}" data-name="${esc(item.name)}" data-kind="iconic">
      <img alt="" src="${item.thumb || ""}">
      <span class="vn">${esc(item.name)}</span>
      <span class="vm">${esc(item.role && item.role !== item.kind ? item.role + " · " + item.kind : item.kind)} · ${esc(item.info)}</span>
    </button>`).join("") || `<div class="hint">還沒有圖像變數</div>`;
  controlList.innerHTML = server.control.map((item) => `
    <button class="var ctrl ${selectedVar === item.name ? "on" : ""}" data-name="${esc(item.name)}" data-kind="control" title="${esc(item.full || item.text)}">
      <span class="vn">${esc(item.name)}<small>${esc(item.type)}</small></span>
      <span class="vm">${esc(item.text)}</span>
    </button>`).join("") || `<div class="hint">還沒有控制變數</div>`;
}

function renderLog() {
  const lines = logHidden ? [] : (server.log || []);
  logBox.innerHTML = lines.map((line) => {
    const cls = String(line).startsWith("錯誤") ? "bad" : "";
    return `<span class="${cls}">${esc(line)}</span>`;
  }).join("\n");
  logBox.scrollTop = logBox.scrollHeight;
  renderProblems();
}

function programFile() {
  return document.getElementById("progName").value || "程式";
}

function noteExprReads(expr, reads) {
  const skip = new Set(["and", "or", "not", "mod", "true", "false"]);
  const re = /[A-Za-z_]\w*/g;
  let match;
  while ((match = re.exec(String(expr)))) {
    if (!skip.has(match[0].toLowerCase())) reads.add(match[0].toLowerCase());
  }
}

function noteCallUses(op, argsBody, lineNo, reads, writes) {
  const args = splitCall(argsBody);
  op.params.forEach((param, index) => {
    const arg = args[index];
    if (arg == null || arg === "") return;
    const ident = arg.match(/^[A-Za-z_]\w*$/);
    if (param.role.includes("輸出")) {
      if (ident) writes.push({ name: ident[0], line: lineNo });
    } else {
      noteExprReads(arg, reads);
    }
  });
}

function collectDiagnostics() {
  const items = [];
  const file = programFile();
  const lines = code.value.split("\n");
  const procedures = new Set();
  lines.forEach((raw) => {
    const header = raw.trim().match(/^procedure\s+([A-Za-z_]\w*)/i);
    if (header) procedures.add(header[1].toLowerCase());
  });
  const reads = new Set();
  const writes = [];
  lines.forEach((raw, index) => {
    const lineNo = index + 1;
    const text = raw.trim();
    if (!text || text.startsWith("*") || text.startsWith("//")) return;
    const loop = text.match(/^for\s+([A-Za-z_]\w*)\s*:=(.*)$/i);
    if (loop) {
      writes.push({ name: loop[1], line: lineNo });
      noteExprReads(loop[2], reads);
      return;
    }
    const dotted = text.match(/^(Vizion\.[A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$/i);
    const bare = text.match(/^([A-Za-z_]\w*)\s*\((.*)\)\s*$/);
    if (dotted) {
      const op = findOperator(dotted[1]);
      if (!op) items.push({ sev: "error", code: "VIZ002", text: `未知算子 ${dotted[1]}`, file, line: lineNo });
      else noteCallUses(op, dotted[2], lineNo, reads, writes);
      return;
    }
    if (bare) {
      const low = bare[1].toLowerCase();
      if (CALL_KEYWORDS.has(low) || procedures.has(low)) {
        noteExprReads(bare[2], reads);
        return;
      }
      const op = findOperator(bare[1]);
      if (op) {
        items.push({ sev: "error", code: "VIZ001", text: `算子要寫成 ${op.name} (...)`, file, line: lineNo });
        noteCallUses(op, bare[2], lineNo, reads, writes);
      } else {
        items.push({ sev: "error", code: "VIZ002", text: `未知算子 ${bare[1]}`, file, line: lineNo });
      }
      return;
    }
    const assign = text.match(/^([A-Za-z_]\w*)\s*:=(.*)$/);
    if (assign) {
      writes.push({ name: assign[1], line: lineNo });
      noteExprReads(assign[2], reads);
    }
  });
  const warned = new Set();
  writes.forEach((item) => {
    const key = item.name.toLowerCase();
    if (warned.has(key) || reads.has(key)) return;
    warned.add(key);
    items.push({ sev: "warning", code: "VIZ100", text: `變數 ${item.name} 指定了但沒有被使用`, file, line: item.line });
  });
  procedures.forEach((name) => {
    const called = lines.some((raw) => new RegExp(`^${name}\\s*\\(`, "i").test(raw.trim()));
    if (called) return;
    const line = lines.findIndex((raw) => new RegExp(`^procedure\\s+${name}\\b`, "i").test(raw.trim())) + 1;
    items.push({ sev: "warning", code: "VIZ101", text: `程序 ${name} 沒有被呼叫`, file, line });
  });
  (server.log || []).forEach((line) => {
    const text = String(line).trim();
    const where = text.match(/第\s*(\d+)\s*行/);
    const lineNo = where ? Number(where[1]) : null;
    if (text.startsWith("錯誤")) {
      if (items.some((item) => item.sev === "error" && item.line === (lineNo || server.errorLine))) return;
      items.push({ sev: "error", code: "VIZ200", text: text.replace(/^錯誤\s*/, ""), file, line: lineNo || server.errorLine });
    } else if (text.startsWith("catch ")) {
      items.push({ sev: "message", code: "VIZ300", text, file, line: lineNo });
    } else if (text.includes("執行完成") || text.startsWith("中斷點") || text.startsWith("已停止") || text.includes("Vizion.stop")) {
      items.push({ sev: "message", code: "VIZ301", text, file, line: lineNo });
    }
  });
  if (server.error && !items.some((item) => item.sev === "error" && item.line === server.errorLine && item.text.includes(server.error))) {
    items.push({ sev: "error", code: "VIZ200", text: server.error, file, line: server.errorLine });
  }
  return items;
}

function renderProblems() {
  const body = document.getElementById("errBody");
  if (!body || !code) return;
  diagnostics = collectDiagnostics();
  const counts = { error: 0, warning: 0, message: 0 };
  diagnostics.forEach((item) => { counts[item.sev] += 1; });
  const errCount = document.getElementById("errCount");
  const warnCount = document.getElementById("warnCount");
  const msgCount = document.getElementById("msgCount");
  if (errCount) errCount.textContent = String(counts.error);
  if (warnCount) warnCount.textContent = String(counts.warning);
  if (msgCount) msgCount.textContent = String(counts.message);
  const badge = document.getElementById("problemBadge");
  if (badge) badge.textContent = String(counts.error || counts.warning || counts.message);
  const stErrors = document.getElementById("stErrors");
  const stWarnings = document.getElementById("stWarnings");
  const stMessages = document.getElementById("stMessages");
  if (stErrors) {
    stErrors.textContent = `錯誤 ${counts.error}`;
    stErrors.classList.toggle("hot-error", counts.error > 0);
  }
  if (stWarnings) {
    stWarnings.textContent = `警告 ${counts.warning}`;
    stWarnings.classList.toggle("hot-warning", counts.warning > 0);
  }
  if (stMessages) stMessages.textContent = `訊息 ${counts.message}`;
  const visible = diagnostics.filter((item) => problemFilter[item.sev]);
  const label = { error: "錯誤", warning: "警告", message: "訊息" };
  body.innerHTML = visible.map((item) => `
    <tr data-line="${item.line || ""}">
      <td class="sev-${item.sev}">${label[item.sev]}</td>
      <td>${esc(item.code)}</td>
      <td class="desc">${esc(item.text)}</td>
      <td>${esc(item.file)}</td>
      <td>${item.line || ""}</td>
    </tr>`).join("") || `<tr><td></td><td></td><td class="desc">沒有要顯示的項目</td><td></td><td></td></tr>`;
}

function goToLine(line) {
  const number = Number(line);
  if (!number) return;
  const lines = code.value.split("\n");
  let pos = 0;
  for (let index = 1; index < number && index <= lines.length; index += 1) pos += lines[index - 1].length + 1;
  code.focus();
  code.selectionStart = pos;
  code.selectionEnd = pos + (lines[number - 1] || "").length;
  revealLine(number);
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
  if (server.line && server.paused === "breakpoint") stLine.textContent = `中斷於第 ${server.line} 行`;
  else if (server.line && server.done) stLine.textContent = `結束於第 ${server.line} 行`;
  else if (server.line) stLine.textContent = `下一行 ${server.line}`;
  else stLine.textContent = server.done ? "程式結束" : "";
}

let imageStamp = null;

function applyState(data) {
  absorbKept(data);
  logHidden = false;
  const nextStamp = data.imageStamp;
  const imageChanged = imageStamp != null && nextStamp != null && nextStamp !== imageStamp;
  if (nextStamp != null) imageStamp = nextStamp;
  previewImg = null;
  previewToken += 1;
  featureLines = [];
  sessionId = data.id;
  server = data;
  gfxBanner.hidden = true;
  renderEditor();
  const arrow = programArrow();
  if (arrow) revealLine(arrow);
  renderVars();
  renderLog();
  setStatus();
  takeDisplay(data);
  if (localStorage.getItem("visdev.contourWidth") == null && data.display) {
    const xld = (data.display.shapes || []).find((shape) => shape.kind === "xld" && shape.width);
    const box = document.getElementById("contourWidth");
    if (xld && box && document.activeElement !== box) box.value = String(contourWidth(xld.width));
  }
  renderHisto(data.histogram || null);
  if (imageChanged) clearPick();
  if (data.error) {
    showTab("problems");
    showErrorDialog(data.error, data.errorLine);
  } else {
    closeErrorDialog();
  }
  publishCustomVars();
  noteRunningProgram();
}

function publishCustomVars() {
  const payload = {
    type: "vars",
    iconic: (server.iconic || []).map((item) => ({ name: item.name, text: `${item.kind} · ${item.info}` })),
    control: (server.control || []).map((item) => ({ name: item.name, type: item.type, text: item.text })),
    at: Date.now(),
  };
  try { localStorage.setItem("visdev.customVars", JSON.stringify(payload)); } catch (_err) { /* 視窗還沒準備 */ }
  try { featureChannel.postMessage(payload); } catch (_err) { /* 頻道稍後才建立 */ }
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

let liveDisplayStamp = null;
let liveDisplayRev = null;
let liveVarRev = null;
let liveLogRev = null;
let liveDisplaySession = null;

function absorbKept(data) {
  if (!server || !data || (data.id && server.id && data.id !== server.id)) return;
  if (data.iconicKept) data.iconic = server.iconic;
  if (data.controlKept) data.control = server.control;
  if (data.logKept) data.log = server.log;
  if (data.displayKept && server.display) data.display = server.display;
}

function takeDisplay(data) {
  if (data.id && data.id !== liveDisplaySession) {
    liveDisplaySession = data.id;
    liveDisplayRev = null;
    liveVarRev = null;
    liveLogRev = null;
  }
  const rev = data.displayRev;
  const same = rev != null && rev === liveDisplayRev;
  if (rev != null) liveDisplayRev = rev;
  if (data.varRev != null) liveVarRev = data.varRev;
  if (data.logRev != null) liveLogRev = data.logRev;
  if (data.displayStamp != null) liveDisplayStamp = data.displayStamp;
  if (data.displayKept || same) return;
  if (data.display) loadDisplay(data.display);
  else clearGraphics();
}

function applyLiveState(data) {
  absorbKept(data);
  const nextStamp = data.imageStamp;
  const imageChanged = imageStamp != null && nextStamp != null && nextStamp !== imageStamp;
  if (nextStamp != null) imageStamp = nextStamp;
  sessionId = data.id;
  server = data;
  renderEditor();
  const arrow = programArrow();
  if (arrow) revealLine(arrow);
  renderLog();
  if (!data.iconicKept && Array.isArray(data.iconic) && Array.isArray(data.control)) renderVars();
  setStatus();
  if (data.running) {
    stState.textContent = "執行中";
    stState.className = "";
  }
  takeDisplay(data);
  if (imageChanged) clearPick();
  noteRunningProgram();
}

async function exec(action, line) {
  if (inflight) return;
  inflight = true;
  document.getElementById("btnRun").disabled = action === "run";
  const submitted = code.value;
  try {
    const res = await fetch("/api/exec", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: sessionId,
        source: submitted,
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
    if (data.running) {
      runningSource = submitted;
      if (data.id !== sessionId) {
        liveDisplayRev = null;
        liveVarRev = null;
        liveLogRev = null;
      }
      sessionId = data.id;
      while (true) {
        const query = new URLSearchParams();
        if (liveDisplayRev != null) query.set("displayRev", String(liveDisplayRev));
        if (liveVarRev != null) query.set("varRev", String(liveVarRev));
        if (liveLogRev != null) query.set("logRev", String(liveLogRev));
        const qs = query.toString();
        const stateRes = await fetch(`/api/session/${encodeURIComponent(sessionId)}/state${qs ? `?${qs}` : ""}`);
        const state = await stateRes.json();
        if (!stateRes.ok) {
          showDetail(state);
          return;
        }
        if (!state.running) {
          applyState(state);
          break;
        }
        applyLiveState(state);
        await new Promise((resolve) => setTimeout(resolve, 120));
      }
      return;
    }
    runningSource = null;
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
  const visible = operators.filter((op) => !query || (op.name + op.purpose + op.principle + op.setup + (op.note || "")).toLowerCase().includes(query));
  let html = "";
  let group = "";
  for (const op of visible) {
    if (op.group !== group) {
      group = op.group;
      html += `<div class="op-group">${esc(group)}</div>`;
    }
    const mark = op.note ? `<i class="op-new">新</i>` : "";
    html += `<button class="op" data-name="${esc(op.name)}" type="button"><b>${esc(op.name)}${mark}</b><span>${esc(op.short)}</span></button>`;
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
    ${noteBlock(op)}
    <h3>用途</h3>
    <p class="lead">${esc(op.purpose)}</p>
    <h3>原理</h3>
    <p class="prose">${esc(op.principle)}</p>
    <h3>如何設定</h3>
    <p class="prose">${esc(op.setup)}</p>
    ${rows ? `<table class="params"><tr><th>參數</th><th>角色</th><th>說明</th></tr>${rows}</table>` : ""}
    <h3>範例用法</h3>
    <div class="hint">${esc(op.prototype)}</div>
    <pre>${esc(op.example)}</pre>
    <button id="btnInsert" type="button">插入範本</button>`;
  document.getElementById("btnInsert").onclick = () => insertAtCursor(commentedCall(op, op.example));
}

function noteBlock(op) {
  if (!op.note) return "";
  return `<h3>註解</h3><p class="prose">${esc(op.note)}</p>`;
}

function commentedCall(op, call) {
  if (!op || !op.note) return call;
  const comment = op.note.split("\n").map((line) => `* ${line}`).join("\n");
  return `${comment}\n${call}`;
}

function insertAtCursor(text) {
  const pos = code.selectionStart;
  const start = code.value.slice(0, pos).lastIndexOf("\n") + 1;
  code.value = code.value.slice(0, start) + text + "\n" + code.value.slice(start);
  const caret = start + text.length + 1;
  code.selectionStart = code.selectionEnd = caret;
  commitProgram("checkpoint");
  renderEditor();
  code.focus();
}

function splitCall(text) {
  return VisdevCall.splitCall(text);
}

function exampleTokens(op) {
  return VisdevCall.exampleTokens(op);
}

const OP_MEMORY = "visdev-op-memory";
const OP_RECENT = "visdev-op-recent";

function readJson(key, fallback) {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "");
    return parsed && typeof parsed === "object" ? parsed : fallback;
  } catch (err) {
    return fallback;
  }
}

function rememberOperator(op, values) {
  const memory = readJson(OP_MEMORY, {});
  memory[op.name] = VisdevCall.memoryRecord(op, values);
  const recent = readJson(OP_RECENT, []).filter((name) => name !== op.name);
  recent.unshift(op.name);
  try {
    localStorage.setItem(OP_MEMORY, JSON.stringify(memory));
    localStorage.setItem(OP_RECENT, JSON.stringify(recent.slice(0, 8)));
  } catch (err) {
    /* the dialog still inserts the call when storage is unavailable */
  }
}

function callValues(op, tokens) {
  return VisdevCall.callValues(op, tokens);
}

let activeOperator = null;
let editingLine = null;

function buildOperatorMenu() {
  const drop = document.getElementById("opMenuDrop");
  if (!drop) return;
  const groups = [];
  for (const op of operators) {
    if (!groups.includes(op.group)) groups.push(op.group);
  }
  const recent = readJson(OP_RECENT, []).filter((name) => operators.some((op) => op.name === name)).slice(0, 8);
  const recentMenu = recent.length ? `<div class="menu-sub"><button type="button" data-submenu="1">最近<span class="menu-arrow">▸</span></button><div class="menu-fly" hidden>${
    recent.map((name) => {
      const op = operators.find((item) => item.name === name);
      return `<button type="button" data-op="${esc(op.name)}"><b>${esc(op.name)}</b><span>${esc(op.short)}</span></button>`;
    }).join("")
  }</div></div>` : "";
  drop.innerHTML = recentMenu + (groups.map((group) => {
    const items = operators.filter((op) => op.group === group).map((op) =>
      `<button type="button" data-op="${esc(op.name)}"><b>${esc(op.name)}</b><span>${esc(op.short)}</span></button>`
    ).join("");
    return `<div class="menu-sub"><button type="button" data-submenu="1">${esc(group)}<span class="menu-arrow">▸</span></button><div class="menu-fly" hidden>${items}</div></div>`;
  }).join("") || `<div class="hint" style="padding:8px">沒有算子</div>`);
}

function findOperator(token) {
  const lower = String(token || "").toLowerCase();
  if (!lower) return null;
  return operators.find((op) => {
    const name = op.name.toLowerCase();
    return name === lower || name === `vizion.${lower}`;
  }) || null;
}

function operatorMatchesPrefix(op, lower) {
  const name = op.name.toLowerCase();
  const bare = name.startsWith("vizion.") ? name.slice("vizion.".length) : name;
  return name.startsWith(lower) || bare.startsWith(lower);
}

function formFields() {
  return [...document.querySelectorAll("#opForm input, #opForm select")];
}

function refreshOpPreview() {
  const preview = document.getElementById("opPreview");
  if (!preview || !activeOperator) return;
  const values = formFields().map((node) => node.value.trim());
  preview.textContent = VisdevCall.formatCall(activeOperator.name, values);
}

function parseOperatorLine(text) {
  return VisdevCall.parseLine(text);
}

function openOperatorAtLine(lineIndex) {
  const parsed = parseOperatorLine(code.value.split("\n")[lineIndex] || "");
  if (!parsed) return;
  const op = findOperator(parsed.name);
  if (!op) return;
  openOperatorForm(op.name, callValues(op, parsed.args), lineIndex);
}

function openOperatorForm(name, preset, lineIndex) {
  const op = operators.find((item) => item.name === name);
  if (!op) return;
  activeOperator = op;
  editingLine = Number.isInteger(lineIndex) ? lineIndex : null;
  showOp(name);
  const saved = readJson(OP_MEMORY, {})[op.name];
  const plan = VisdevCall.dialogValues(op, { preset: preset || null, lineIndex: editingLine, memory: saved });
  const values = plan.values;
  const ok = document.getElementById("opDialogOk");
  if (ok) ok.textContent = plan.label;
  document.getElementById("opDialogTitle").textContent = op.name;
  document.getElementById("opDialogGuide").innerHTML = `
    ${noteBlock(op)}
    <h3>用途</h3>
    <p class="guide-body">${esc(op.purpose)}</p>
    <h3>原理</h3>
    <p class="guide-body">${esc(op.principle)}</p>
    <h3>如何設定</h3>
    <p class="guide-body">${esc(op.setup)}</p>`;
  let role = "";
  let html = "";
  op.params.forEach((param, index) => {
    if (param.role !== role) {
      role = param.role;
      html += `<h3>${esc(role)}</h3>`;
    }
    const choices = param.choices || [];
    const current = values[index];
    let control = `<input data-index="${index}" value="${esc(current)}" spellcheck="false">`;
    if (param.role === "控制輸入" && choices.length) {
      const known = choices.includes(current);
      const extras = !known && current !== undefined && current !== "" ? [current] : [];
      const picked = known ? current : extras[0] || choices[0];
      const options = extras.concat(choices).map((choice) => {
        const label = String(choice).replace(/^'(.*)'$/, "$1");
        return `<option value="${esc(choice)}"${choice === picked ? " selected" : ""}>${esc(label)}</option>`;
      }).join("");
      control = `<select data-index="${index}">${options}</select>`;
    }
    html += `<label class="op-field"><span>${esc(param.name)}</span>${control}<em>${esc(param.desc)}</em></label>`;
  });
  document.getElementById("opForm").innerHTML = html || `<p class="hint">這個算子沒有參數。</p>`;
  document.getElementById("opDialog").hidden = false;
  refreshShapeTools();
  refreshOpPreview();
  document.querySelector("#opForm input, #opForm select")?.focus();
}

function closeOperatorDialog() {
  document.getElementById("opDialog").hidden = true;
  editingLine = null;
  const ok = document.getElementById("opDialogOk");
  if (ok) ok.textContent = "插入";
  cancelShapeDraw();
  refreshShapeTools();
}

function insertOperatorCall() {
  if (!activeOperator) return;
  const inputs = formFields();
  for (const input of inputs) {
    const param = activeOperator.params[Number(input.dataset.index)];
    const value = input.value.trim();
    if (!value) {
      showErrorDialog(`${param.name} 還沒有填`);
      input.focus();
      return;
    }
    if (param.role !== "控制輸入" && !/^[A-Za-z_][A-Za-z0-9_]*$/.test(value)) {
      showErrorDialog(`${param.name} 必須是變數名稱`);
      input.focus();
      return;
    }
  }
  const values = inputs.map((node) => node.value.trim());
  const call = VisdevCall.formatCall(activeOperator.name, values);
  rememberOperator(activeOperator, values);
  if (editingLine != null) {
    const replaced = VisdevCall.replaceLine(code.value, editingLine, call);
    code.value = replaced.text;
    code.selectionStart = code.selectionEnd = replaced.caret;
    commitProgram("checkpoint");
    renderEditor();
  } else {
    insertAtCursor(commentedCall(activeOperator, call));
  }
  buildOperatorMenu();
  closeOperatorDialog();
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
  return VisdevCall.exampleTokens({ example: example });
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
  const tokenMatch = chunk.match(/(?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*$/i);
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
    const call = raw.match(/^((?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$/i);
    if (!call) continue;
    const op = findOperator(call[1]);
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
    const nameMatch = ctx.paren.before.match(/(?:^|\s)((?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*)\s*$/i);
    const opName = nameMatch ? nameMatch[1] : "";
    const op = findOperator(opName);
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
    .filter((op) => operatorMatchesPrefix(op, lower))
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
  commitProgram("checkpoint");
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
  publishDisplay(null);
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
  publishDisplay(display);
  gfxEmpty.hidden = true;
  gfxTitle.textContent = `${display.width} × ${display.height}`;
  const refit = VisdevGeom.shouldRefit(cam.user, cam.fitW, cam.fitH, display.width, display.height);
  cam.fitW = display.width;
  cam.fitH = display.height;
  if (refit) {
    cam.user = false;
    fit(false);
  } else {
    draw();
  }
}

function fit(markUser = false) {
  const rect = canvas.getBoundingClientRect();
  if (!disp || !disp.width || rect.width < 2) return;
  const placed = VisdevGeom.fitView(rect.width, rect.height, disp.width, disp.height, 18);
  cam.scale = placed.scale;
  cam.x = placed.x;
  cam.y = placed.y;
  cam.user = markUser;
  draw();
}

function zoomBy(factor) {
  if (!disp) return;
  const rect = canvas.getBoundingClientRect();
  const placed = VisdevGeom.zoomAt(cam, rect.width / 2, rect.height / 2, factor);
  cam.scale = placed.scale;
  cam.x = placed.x;
  cam.y = placed.y;
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

function strokePicked(ctx) {
  if (!pickedObject || !disp) return;
  const rings = (disp.shapes || []).filter((shape) => {
    if (shape.type !== "polyline" || shape.points?.length < 2) return false;
    if (shape.source !== pickedObject.name) return false;
    if ((shape.kind || "region") !== pickedObject.kind) return false;
    if (pickedObject.group != null && shape.group != null) return shape.group === pickedObject.group;
    return shape.object === pickedObject.index;
  });
  if (!rings.length) return;
  const drawRings = () => {
    rings.forEach((shape) => {
      ctx.beginPath();
      ctx.moveTo(shape.points[0][0], shape.points[0][1]);
      for (let i = 1; i < shape.points.length; i += 1) ctx.lineTo(shape.points[i][0], shape.points[i][1]);
      if (shape.closed !== false) ctx.closePath();
      ctx.stroke();
    });
  };
  ctx.save();
  ctx.setLineDash([7 / cam.scale, 5 / cam.scale]);
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(0,0,0,0.85)";
  ctx.lineWidth = 4 / cam.scale;
  drawRings();
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = 2 / cam.scale;
  drawRings();
  ctx.restore();
}

function contourWidth(fallback) {
  const stored = localStorage.getItem("visdev.contourWidth");
  const raw = stored == null || stored === "" ? Number(fallback) : Number(stored);
  const value = Number.isFinite(raw) ? raw : 2;
  return Math.min(10, Math.max(1, Math.round(value)));
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
    if (shapeDraw && shapeDraw.a && shapeDraw.b) {
      ctx.save();
      ctx.translate(cam.x, cam.y);
      ctx.scale(cam.scale, cam.scale);
      strokeDraft(ctx);
      ctx.restore();
    }
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
    if (shape.type === "polyline" && shape.points?.length && !shape.hit) {
      ctx.beginPath();
      ctx.moveTo(shape.points[0][0], shape.points[0][1]);
      for (let i = 1; i < shape.points.length; i += 1) ctx.lineTo(shape.points[i][0], shape.points[i][1]);
      if (shape.closed) ctx.closePath();
      ctx.strokeStyle = shape.color || "#00e05a";
      const width = shape.kind === "xld" ? contourWidth(shape.width || 2) : (shape.width || 2);
      ctx.lineWidth = width / cam.scale;
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
    ctx.lineWidth = contourWidth(2) / cam.scale;
    for (const line of featureLines) {
      if (!line || line.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(line[0][0], line[0][1]);
      for (let i = 1; i < line.length; i += 1) ctx.lineTo(line[i][0], line[i][1]);
      ctx.stroke();
    }
  }
  strokePicked(ctx);
  strokeDraft(ctx);
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
  if (zoomBox) {
    const x = Math.min(zoomBox.x, zoomBox.x2);
    const y = Math.min(zoomBox.y, zoomBox.y2);
    const w = Math.abs(zoomBox.x2 - zoomBox.x);
    const h = Math.abs(zoomBox.y2 - zoomBox.y);
    ctx.fillStyle = "rgba(255, 225, 74, 0.12)";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "#ffe14a";
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
  }
  stZoom.textContent = `${Math.round(cam.scale * 100)}%`;
}

function imagePoint(event) {
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const image = VisdevGeom.imagePoint(x, y, cam);
  return { x, y, col: image.col, row: image.row };
}

const probeBox = document.getElementById("probe");

function showProbe(point, lines) {
  probeBox.innerHTML = lines.map((line) => esc(line)).join("<br>");
  probeBox.hidden = false;
  let left = point.x + 16;
  let top = point.y + 18;
  const boxW = probeBox.offsetWidth;
  const boxH = probeBox.offsetHeight;
  if (left + boxW + 8 > canvas.clientWidth) left = point.x - boxW - 16;
  if (top + boxH + 8 > canvas.clientHeight) top = point.y - boxH - 16;
  probeBox.style.left = `${Math.max(4, left)}px`;
  probeBox.style.top = `${Math.max(4, top)}px`;
}

function hideProbe() {
  probeBox.hidden = true;
}

function updateProbe(event) {
  if (!disp) {
    stPix.textContent = "Row —   Column —   R —  G —  B —";
    mouse = null;
    hideProbe();
    return;
  }
  const point = imagePoint(event);
  mouse = point;
  const col = Math.floor(point.col);
  const row = Math.floor(point.row);
  const where = `Row ${point.row.toFixed(1)}   Column ${point.col.toFixed(1)}`;
  let color = "R —  G —  B —";
  if (col >= 0 && row >= 0 && col < disp.width && row < disp.height && baseImage) {
    const pixel = probeCtx.getImageData(col, row, 1, 1).data;
    color = `R ${pixel[0]}  G ${pixel[1]}  B ${pixel[2]}`;
  }
  stPix.textContent = `${where}   ${color}`;
  showProbe(point, [where, color]);
  draw();
}

function useNewImage(path) {
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const cursorLine = code.value.slice(0, code.selectionStart).split("\n").length - 1;
  let index = /^\s*read_image\s*\(/i.test(lines[cursorLine] || "") ? cursorLine : lines.findIndex((line) => /^\s*read_image\s*\(/i.test(line));
  if (index >= 0) {
    const match = lines[index].match(/^(\s*)read_image\s*\(\s*([^,\s]+)\s*,/i);
    const indent = match ? match[1] : "";
    const name = match ? match[2] : "Image";
    lines[index] = `${indent}read_image (${name}, '${path}')`;
    code.value = lines.join("\n");
    commitProgram("checkpoint");
    renderEditor();
    return;
  }
  insertAtCursor(`read_image (Image, '${path}')`);
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
  useNewImage(data.path);
  await exec("run");
}

window.addEventListener("visdev-layout", () => resizeCanvas());

code.addEventListener("input", () => {
  if (!composing) commitProgram("type");
  refreshDirty();
  renderEditor();
  updateSense();
  noteRunningProgram();
});
document.getElementById("btnClearLog").addEventListener("click", () => {
  logHidden = true;
  logBox.textContent = "";
});
document.getElementById("errFilters").addEventListener("click", (event) => {
  const button = event.target.closest(".errfilter");
  if (!button) return;
  const sev = button.dataset.sev;
  problemFilter[sev] = !problemFilter[sev];
  button.classList.toggle("on", problemFilter[sev]);
  renderProblems();
});
document.getElementById("errBody").addEventListener("click", (event) => {
  const row = event.target.closest("tr");
  if (!row || !row.dataset.line) return;
  goToLine(row.dataset.line);
});
document.getElementById("stErrors").addEventListener("click", () => showTab("problems"));
document.getElementById("stWarnings").addEventListener("click", () => showTab("problems"));
document.getElementById("stMessages").addEventListener("click", () => showTab("problems"));
code.addEventListener("scroll", () => {
  hl.scrollTop = code.scrollTop;
  hl.scrollLeft = code.scrollLeft;
  gutter.scrollTop = code.scrollTop;
  lineTimes.scrollTop = code.scrollTop;
  placeBars();
  placeSense();
});
code.addEventListener("keydown", (event) => {
  if (!composing && !event.isComposing && (event.ctrlKey || event.metaKey) && !event.altKey) {
    const key = event.key.toLowerCase();
    if (key === "z" || key === "y") {
      event.preventDefault();
      applyHistoryEntry((key === "y" || event.shiftKey) ? codeHistory.redo() : codeHistory.undo());
      return;
    }
  }
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
code.addEventListener("click", () => {
  updateSense();
  if (code.selectionStart !== code.selectionEnd) return;
  const pos = code.selectionStart;
  const lineIndex = code.value.slice(0, pos).split("\n").length - 1;
  const line = code.value.split("\n")[lineIndex] || "";
  const parsed = parseOperatorLine(line);
  if (!parsed) return;
  const nameAt = line.indexOf(parsed.name);
  if (nameAt < 0) return;
  const col = pos - (code.value.lastIndexOf("\n", Math.max(0, pos - 1)) + 1);
  if (col >= nameAt && col <= nameAt + parsed.name.length) openOperatorAtLine(lineIndex);
});
code.addEventListener("dblclick", (event) => {
  const lines = code.value.split("\n");
  const style = getComputedStyle(code);
  const lineHeight = parseFloat(style.lineHeight) || 20;
  const pad = parseFloat(style.paddingTop) || 0;
  const fromY = Math.floor((event.offsetY + code.scrollTop - pad) / lineHeight);
  const fromCaret = code.value.slice(0, code.selectionStart).split("\n").length - 1;
  for (const line of [fromY, fromCaret]) {
    if (line >= 0 && line < lines.length && parseOperatorLine(lines[line])) {
      openOperatorAtLine(line);
      return;
    }
  }
});
code.addEventListener("compositionstart", () => {
  composing = true;
  codeHistory.beginComposition();
  closeSense();
});
code.addEventListener("compositionend", () => {
  composing = false;
  codeHistory.endComposition(code.value, code.selectionStart, code.selectionEnd);
  updateSense();
});
code.addEventListener("blur", () => {
  setTimeout(() => {
    if (document.activeElement !== code) closeSense();
  }, 150);
});
let arrowDrag = null;
let skipGutterClick = false;

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
  skipGutterClick = true;
  setTimeout(() => { skipGutterClick = false; }, 0);
  document.body.classList.remove("arrow-drag");
  gutter.querySelectorAll(".gl.aim").forEach((node) => node.classList.remove("aim"));
  if (target && target !== from) exec("goto", target);
});
gutter.addEventListener("click", (event) => {
  if (skipGutterClick) {
    skipGutterClick = false;
    return;
  }
  const lineNode = event.target.closest(".gl");
  if (!lineNode) return;
  const line = Number(lineNode.dataset.line);
  if (breakpoints.has(line)) breakpoints.delete(line);
  else breakpoints.add(line);
  renderGutter();
  pushBreakpoints();
});
document.getElementById("errOk").onclick = () => closeErrorDialog();
document.getElementById("errDialog").addEventListener("mousedown", (event) => {
  if (event.target.id === "errDialog") closeErrorDialog();
});
window.addEventListener("keydown", (event) => {
  if (event.key !== "Escape") return;
  if (!document.getElementById("errDialog").hidden) {
    event.preventDefault();
    closeErrorDialog();
  }
  if (!document.getElementById("aboutDialog").hidden) {
    event.preventDefault();
    document.getElementById("aboutDialog").hidden = true;
  }
  if (shapeDraw) {
    event.preventDefault();
    cancelShapeDraw();
    return;
  }
  if (!document.getElementById("opDialog").hidden) {
    event.preventDefault();
    closeOperatorDialog();
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
  const placed = VisdevGeom.oneToOne(rect.width, rect.height, disp.width, disp.height);
  cam.scale = placed.scale;
  cam.x = placed.x;
  cam.y = placed.y;
  cam.user = true;
  draw();
};
document.getElementById("btnZoomIn").onclick = () => zoomBy(1.25);
const contourWidthInput = document.getElementById("contourWidth");
contourWidthInput.value = String(contourWidth());
contourWidthInput.addEventListener("change", () => {
  const value = Math.min(10, Math.max(1, Math.round(Number(contourWidthInput.value) || 1)));
  contourWidthInput.value = String(value);
  localStorage.setItem("visdev.contourWidth", String(value));
  draw();
});
window.addEventListener("storage", (event) => {
  if (event.key !== "visdev.contourWidth") return;
  if (document.activeElement === contourWidthInput) return;
  contourWidthInput.value = String(contourWidth());
  draw();
});
document.getElementById("btnZoomOut").onclick = () => zoomBy(0.8);
document.getElementById("btnUpload").onclick = () => document.getElementById("fileImage").click();
document.getElementById("fileImage").onchange = () => {
  const file = document.getElementById("fileImage").files[0];
  if (file) uploadFile(file);
};
document.getElementById("btnOpen").onclick = () => document.getElementById("fileOpen").click();
document.getElementById("fileOpen").onchange = async () => {
  const input = document.getElementById("fileOpen");
  const file = input.files[0];
  input.value = "";
  if (file) await openLocalFile(file);
};

function isHalconProgram(name) {
  const lower = String(name || "").toLowerCase();
  return [".hdev", ".hdvp", ".hdpl", ".dev", ".dvp"].some((suffix) => lower.endsWith(suffix));
}

async function openLocalFile(file) {
  let source = "";
  let name = file.name;
  let note = `已開啟 ${name}`;
  if (isHalconProgram(file.name)) {
    const body = new FormData();
    body.append("file", file);
    const res = await fetch("/api/import/hdevelop", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) {
      showDetail(data);
      return;
    }
    source = data.source || "";
    name = data.name || name;
    if (data.skipped && data.skipped.length) {
      const names = data.skipped;
      note = names.length > 4
        ? `已開啟 ${name}。${names.length} 個有密碼的程序沒有打開`
        : `已開啟。有密碼的程序沒有打開：${names.join("、")}`;
    } else {
      note = `已開啟 ${name}`;
    }
  } else {
    source = await file.text();
  }
  code.value = source.replace(/\r\n/g, "\n");
  document.getElementById("progName").value = name;
  saveTarget = { kind: "import", name };
  sessionId = null;
  breakpoints = new Set();
  resetProgramHistory();
  markSaved();
  renderEditor();
  await exec("reset");
  stState.textContent = note;
  stState.className = "ok";
}
function programFileName(name) {
  const trimmed = String(name || "").trim() || "program.viz";
  const lower = trimmed.toLowerCase();
  return lower.endsWith(".hdv") || lower.endsWith(".viz") ? trimmed : `${trimmed}.viz`;
}

function desktopApi() {
  return window.pywebview && window.pywebview.api;
}

async function saveToWorkspace(name, copy) {
  const res = await fetch("/api/programs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, source: code.value }),
  });
  const data = await res.json();
  if (!res.ok) {
    showDetail(data);
    return false;
  }
  document.getElementById("progName").value = data.name;
  saveTarget = { kind: "workspace", name: data.name };
  markSaved();
  stState.textContent = copy ? `範例已另存 ${data.path}` : `已儲存 ${data.path}`;
  stState.className = "ok";
  loadFiles();
  return true;
}

async function rememberFileSave(data) {
  document.getElementById("progName").value = data.name;
  saveTarget = { kind: "file", name: data.name, path: data.path };
  markSaved();
  stState.textContent = `已儲存 ${data.path}`;
  stState.className = "ok";
}

async function saveCurrent(saveAs) {
  const plan = window.VisdevProgram.savePlan(saveTarget, !!saveAs);
  const api = desktopApi();
  if (plan.mode === "file" && api && api.write_program_path) {
    try {
      await rememberFileSave(await api.write_program_path(plan.path, code.value));
    } catch (err) {
      showDetail({ error: "存檔失敗：" + err.message });
    }
    return;
  }
  if (plan.mode === "ask") {
    if (api && api.save_program_dialog) {
      try {
        const data = await api.save_program_dialog(programFileName(plan.name), code.value);
        if (!data || data.cancelled) return;
        if (data.error) {
          showDetail({ error: data.error });
          return;
        }
        await rememberFileSave(data);
      } catch (err) {
        showDetail({ error: "存檔失敗：" + err.message });
      }
      return;
    }
    const name = window.prompt("另存新檔的檔名", programFileName(plan.name));
    if (!name || !name.trim()) return;
    await saveToWorkspace(programFileName(name), false);
    return;
  }
  await saveToWorkspace(programFileName(plan.name), !!plan.copy);
}

document.getElementById("btnSave").onclick = () => saveCurrent(false);
document.getElementById("progName").addEventListener("input", refreshDirty);
function placeSampleMenu(anchor) {
  sampleMenu.hidden = false;
  if (!anchor) {
    sampleMenu.style.position = "";
    sampleMenu.style.left = "";
    sampleMenu.style.top = "";
    sampleMenu.style.zIndex = "";
    return;
  }
  const rect = anchor.getBoundingClientRect();
  sampleMenu.style.position = "fixed";
  sampleMenu.style.zIndex = "80";
  sampleMenu.style.left = `${rect.right - 4}px`;
  sampleMenu.style.top = `${rect.top}px`;
  const box = sampleMenu.getBoundingClientRect();
  if (rect.right + box.width > window.innerWidth - 8) {
    sampleMenu.style.left = `${Math.max(8, rect.left - box.width)}px`;
  }
  if (rect.top + box.height > window.innerHeight - 8) {
    sampleMenu.style.top = `${Math.max(8, window.innerHeight - box.height - 8)}px`;
  }
}

document.getElementById("btnSamples").onclick = (event) => {
  event.stopPropagation();
  if (!sampleMenu.hidden) {
    sampleMenu.hidden = true;
    return;
  }
  placeSampleMenu(null);
};
document.addEventListener("click", (event) => {
  if (event.target.closest(".pop") || event.target.closest(".menu") || event.target.closest("[data-cmd='samples']")) return;
  sampleMenu.hidden = true;
});

function editCode(command) {
  if (command === "undo") {
    applyHistoryEntry(codeHistory.undo());
    code.focus();
    return;
  }
  if (command === "redo") {
    applyHistoryEntry(codeHistory.redo());
    code.focus();
    return;
  }
  code.focus();
  document.execCommand(command);
  commitProgram("checkpoint");
  renderEditor();
}

function syncViewMenu() {
  document.querySelectorAll(".menu-drop button[data-pane]").forEach((button) => {
    const hidden = window.VisDevDock && VisDevDock.isAutoHide(button.dataset.pane);
    button.setAttribute("aria-checked", hidden ? "false" : "true");
  });
  const trace = document.getElementById("menuTrace");
  if (trace) trace.setAttribute("aria-checked", document.getElementById("trace").checked ? "true" : "false");
}

window.VisDevCommands = {
  new() {
    code.value = "* 新程式\n";
    document.getElementById("progName").value = "untitled.viz";
    saveTarget = { kind: "workspace", name: "untitled.viz" };
    savedText = "";
    sessionId = null;
    breakpoints = new Set();
    resetProgramHistory();
    refreshDirty();
    renderEditor();
    exec("reset");
  },
  async open() {
    const api = desktopApi();
    if (api && api.open_program_dialog) {
      try {
        const picked = await api.open_program_dialog();
        if (!picked || picked.cancelled) return;
        if (picked.error) {
          showDetail({ error: picked.error });
          return;
        }
        let note = `已開啟 ${picked.name}`;
        if (picked.skipped && picked.skipped.length) {
          note = `已開啟 ${picked.name}。${picked.skipped.length} 個有密碼的程序沒有打開`;
        }
        code.value = String(picked.source || "").replace(/\r\n/g, "\n");
        document.getElementById("progName").value = picked.name;
        saveTarget = picked.path
          ? { kind: "file", name: picked.name, path: picked.path }
          : { kind: "import", name: picked.name };
        sessionId = null;
        breakpoints = new Set();
        resetProgramHistory();
        markSaved();
        renderEditor();
        await exec("reset");
        stState.textContent = note;
        stState.className = "ok";
        return;
      } catch (_err) {
        /* 沒有桌面對話框時改用檔案選擇 */
      }
    }
    document.getElementById("fileOpen").click();
  },
  save() { saveCurrent(false); },
  saveAs() { saveCurrent(true); },
  loadImage() { document.getElementById("fileImage").click(); },
  samples(anchor) { placeSampleMenu(anchor); },
  openSample(name) { openProgram("samples", name); },
  undo() { editCode("undo"); },
  redo() { editCode("redo"); },
  cut() { editCode("cut"); },
  copy() { editCode("copy"); },
  paste() { editCode("paste"); },
  selectAll() {
    code.focus();
    code.setSelectionRange(0, code.value.length);
  },
  fit() { document.getElementById("btnFit").click(); },
  one() { document.getElementById("btnOne").click(); },
  zoomIn() { document.getElementById("btnZoomIn").click(); },
  zoomOut() { document.getElementById("btnZoomOut").click(); },
  trace() {
    const box = document.getElementById("trace");
    box.checked = !box.checked;
    syncViewMenu();
  },
  run() { exec("run"); },
  step() { exec("step"); },
  stop() { document.getElementById("btnStop").click(); },
  reset() { exec("reset"); },
  threshold() { openThresholdWindow(); },
  subpix() { openBinWindow(); },
  features() { openFeatureWindow(); },
  ocr() { applyOcr(); },
  connection() {
    applyConnection(document.getElementById("connMin").value, document.getElementById("connDist").value);
  },
  layout() { if (window.VisDevDock) VisDevDock.reset(); },
  floatGfx() { openGfxWindow(); },
  about() { document.getElementById("aboutDialog").hidden = false; },
  openOperator(name) { openOperatorForm(name); },
};

window.addEventListener("visdev-layout", syncViewMenu);
syncViewMenu();
document.getElementById("aboutOk").onclick = () => { document.getElementById("aboutDialog").hidden = true; };
document.getElementById("opDialogCancel").onclick = () => closeOperatorDialog();
document.getElementById("opDialogOk").onclick = () => insertOperatorCall();
document.getElementById("opDialog").addEventListener("keydown", (event) => {
  if (event.key !== "Enter" || event.shiftKey || event.altKey || event.ctrlKey || event.metaKey || event.isComposing) return;
  event.preventDefault();
  insertOperatorCall();
});
document.getElementById("opForm").addEventListener("input", refreshOpPreview);
document.getElementById("opForm").addEventListener("change", refreshOpPreview);

window.addEventListener("keydown", (event) => {
  if (event.key === "F5") { event.preventDefault(); exec("run"); }
  if (event.key === "F6") { event.preventDefault(); exec("step"); }
  if (event.key === "F7") { event.preventDefault(); exec("reset"); }
  if (!(event.ctrlKey || event.metaKey) || event.altKey) return;
  const key = event.key.toLowerCase();
  if (key === "s") { event.preventDefault(); VisDevCommands.save(); }
  if (key === "o") { event.preventDefault(); VisDevCommands.open(); }
  if (key === "n") { event.preventDefault(); VisDevCommands.new(); }
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
  if (op) insertAtCursor(commentedCall(op, op.example));
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
  clearPick();
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
  const placed = VisdevGeom.zoomAt(cam, point.x, point.y, event.deltaY < 0 ? 1.12 : 1 / 1.12);
  cam.x = placed.x;
  cam.y = placed.y;
  cam.scale = placed.scale;
  cam.user = true;
  draw();
}, { passive: false });

const mouseModes = ["select", "pan", "zoom"];
let mouseMode = "pan";
let dragging = null;
let zoomBox = null;

function applyMouseMode(mode) {
  if (!mouseModes.includes(mode)) mode = "select";
  mouseMode = mode;
  canvas.classList.remove("mode-select", "mode-pan", "mode-zoom", "dragging");
  canvas.classList.add(`mode-${mode}`);
  document.querySelectorAll("#gfxMouse button").forEach((button) => {
    const on = button.dataset.mode === mode;
    button.classList.toggle("on", on);
    button.setAttribute("aria-pressed", on ? "true" : "false");
  });
  try { localStorage.setItem("visdev.gfxMouse", mode); } catch (_err) { /* 視窗還沒準備 */ }
}

function zoomAtScreen(sx, sy, factor) {
  const placed = VisdevGeom.zoomAt(cam, sx, sy, factor);
  cam.scale = placed.scale;
  cam.x = placed.x;
  cam.y = placed.y;
  cam.user = true;
}

function zoomToRect(col0, row0, col1, row1) {
  const rect = canvas.getBoundingClientRect();
  const placed = VisdevGeom.zoomRect(rect.width, rect.height, col0, row0, col1, row1, 8);
  cam.scale = placed.scale;
  cam.x = placed.x;
  cam.y = placed.y;
  cam.user = true;
}

applyMouseMode(mouseModes.includes(localStorage.getItem("visdev.gfxMouse")) ? localStorage.getItem("visdev.gfxMouse") : "select");
document.getElementById("gfxMouse").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-mode]");
  if (button) applyMouseMode(button.dataset.mode);
});

const drawMenu = document.getElementById("drawMenu");
const shapeConfirm = document.getElementById("shapeConfirm");
const shapeConfirmText = document.getElementById("shapeConfirmText");
let shapeDraw = null;

function fmtGeom(value) {
  const rounded = Math.round(Number(value) * 100) / 100;
  return String(rounded);
}

function geometryInputs() {
  if (!activeOperator || document.getElementById("opDialog").hidden) return {};
  const names = new Set(["Row", "Column", "Row1", "Column1", "Row2", "Column2", "Radius", "Radius1", "Radius2", "Phi", "Length1", "Length2"]);
  const map = {};
  formFields().forEach((node) => {
    const param = activeOperator.params[Number(node.dataset.index)];
    if (!param || param.role !== "控制輸入" || !names.has(param.name) || node.tagName === "SELECT") return;
    map[param.name] = node;
  });
  return map;
}

function refreshShapeTools() {
  const box = document.getElementById("shapeTools");
  const hint = document.getElementById("opDrawHint");
  const tools = geometryTools(geometryInputs());
  if (hint) hint.hidden = tools.length === 0;
  if (!box) return;
  box.hidden = tools.length === 0;
  box.querySelectorAll("button[data-draw]").forEach((button) => {
    const on = tools.includes(button.dataset.draw);
    button.hidden = !on;
    button.classList.toggle("on", on && shapeDraw && shapeDraw.tool === button.dataset.draw);
  });
}

function geometryTools(map) {
  return VisdevGeom.geometryTools(Object.keys(map));
}

function hideDrawMenu() {
  if (drawMenu) drawMenu.hidden = true;
}

function cancelShapeDraw() {
  shapeDraw = null;
  hideDrawMenu();
  if (shapeConfirm) shapeConfirm.hidden = true;
  draw();
  refreshShapeTools();
}

function draftBox(draw) {
  return VisdevGeom.draftBox(draw.a, draw.b);
}

function strokeDraft(pen) {
  if (!shapeDraw || !shapeDraw.a || !shapeDraw.b) return;
  const draw = shapeDraw;
  pen.save();
  pen.strokeStyle = "#ffe14a";
  pen.lineWidth = 2 / cam.scale;
  pen.setLineDash([6 / cam.scale, 4 / cam.scale]);
  if (draw.tool === "circle") {
    pen.beginPath();
    pen.arc(draw.a.col, draw.a.row, Math.hypot(draw.b.col - draw.a.col, draw.b.row - draw.a.row), 0, Math.PI * 2);
    pen.stroke();
  } else if (draw.tool === "ellipse") {
    const box = draftBox(draw);
    pen.beginPath();
    pen.ellipse(box.col1 + box.width / 2, box.row1 + box.height / 2, Math.max(box.width / 2, 0.5), Math.max(box.height / 2, 0.5), 0, 0, Math.PI * 2);
    pen.stroke();
  } else {
    const box = draftBox(draw);
    pen.strokeRect(box.col1, box.row1, box.width, box.height);
  }
  pen.restore();
}

function showShapeConfirm() {
  const map = geometryInputs();
  const box = draftBox(shapeDraw);
  let text = "";
  if (shapeDraw.tool === "circle") {
    const radius = Math.hypot(shapeDraw.b.col - shapeDraw.a.col, shapeDraw.b.row - shapeDraw.a.row);
    text = `圓  列 ${fmtGeom(shapeDraw.a.row)}  行 ${fmtGeom(shapeDraw.a.col)}  半徑 ${fmtGeom(radius)}`;
  } else if (shapeDraw.tool === "ellipse") {
    text = `橢圓  列 ${fmtGeom((box.row1 + box.row2) / 2)}  行 ${fmtGeom((box.col1 + box.col2) / 2)}  半徑 ${fmtGeom(box.width / 2)} × ${fmtGeom(box.height / 2)}`;
  } else if (map.Row1) {
    text = `矩形  列 ${fmtGeom(box.row1)}–${fmtGeom(box.row2)}  行 ${fmtGeom(box.col1)}–${fmtGeom(box.col2)}`;
  } else {
    text = `矩形  列 ${fmtGeom((box.row1 + box.row2) / 2)}  行 ${fmtGeom((box.col1 + box.col2) / 2)}  半長 ${fmtGeom(box.width / 2)} × ${fmtGeom(box.height / 2)}`;
  }
  shapeConfirmText.textContent = text;
  shapeConfirm.hidden = false;
}

function writeGeom(map, name, value) {
  if (!map[name]) return;
  map[name].value = fmtGeom(value);
}

function clearUnusedGeom(map, keep) {
  Object.keys(map).forEach((name) => {
    if (keep.has(name)) return;
    const param = activeOperator.params.find((item) => item.role === "控制輸入" && item.name === name);
    if (!param || (param.default !== "''" && param.default !== "")) return;
    map[name].value = "''";
  });
}

function applyShapeDraw() {
  if (!shapeDraw || !shapeDraw.a || !shapeDraw.b) return;
  const map = geometryInputs();
  const inner = map.Radius1 && (activeOperator.params.find((param) => param.name === "Radius1")?.desc || "").includes("內");
  const mode = shapeDraw.tool === "rect" && map.Row1 ? "corners" : (inner ? "inner" : "axis");
  const numbers = VisdevGeom.shapeNumbers(shapeDraw.tool, shapeDraw.a, shapeDraw.b, mode);
  Object.keys(numbers).forEach((name) => writeGeom(map, name, numbers[name]));
  const keep = new Set(Object.keys(numbers));
  clearUnusedGeom(map, keep);
  refreshOpPreview();
  cancelShapeDraw();
}

function showDrawMenu(x, y) {
  const map = geometryInputs();
  const tools = geometryTools(map);
  if (!tools.length) {
    hideDrawMenu();
    return;
  }
  const labels = { rect: "矩形", circle: "圓", ellipse: "橢圓" };
  drawMenu.innerHTML = tools.map((tool) => `<button type="button" data-draw="${tool}">${labels[tool]}</button>`).join("");
  drawMenu.hidden = false;
  const width = 120;
  const height = tools.length * 28 + 4;
  drawMenu.style.left = `${Math.max(8, Math.min(x, window.innerWidth - width - 8))}px`;
  drawMenu.style.top = `${Math.max(8, Math.min(y, window.innerHeight - height - 8))}px`;
}

function beginShapeDraw(tool) {
  shapeDraw = { tool, a: null, b: null, pending: false };
  shapeConfirm.hidden = true;
  hideDrawMenu();
  refreshShapeTools();
  draw();
}

drawMenu.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-draw]");
  if (!button) return;
  beginShapeDraw(button.dataset.draw);
});
document.getElementById("shapeTools").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-draw]");
  if (button) beginShapeDraw(button.dataset.draw);
});
document.getElementById("btnFloatGfx").onclick = () => openGfxWindow();
document.getElementById("shapeOk").onclick = () => applyShapeDraw();
document.getElementById("shapeCancel").onclick = () => cancelShapeDraw();
document.addEventListener("mousedown", (event) => {
  if (drawMenu.hidden) return;
  if (event.target.closest("#drawMenu")) return;
  hideDrawMenu();
});

canvas.addEventListener("contextmenu", (event) => {
  const map = geometryInputs();
  if (!geometryTools(map).length) return;
  event.preventDefault();
  showDrawMenu(event.clientX, event.clientY);
});

canvas.addEventListener("mousedown", (event) => {
  if (shapeDraw && event.button === 0) {
    const point = imagePoint(event);
    shapeDraw.a = { col: point.col, row: point.row };
    shapeDraw.b = { col: point.col, row: point.row };
    shapeDraw.pending = false;
    shapeConfirm.hidden = true;
    hideDrawMenu();
    draw();
    return;
  }
  if (event.button !== 0 || !disp) return;
  const point = imagePoint(event);
  dragging = { x: point.x, y: point.y, camX: cam.x, camY: cam.y, col: point.col, row: point.row, mode: mouseMode };
  if (mouseMode === "pan") canvas.classList.add("dragging");
});
window.addEventListener("mouseup", (event) => {
  if (shapeDraw && shapeDraw.a && event.button === 0) {
    const point = imagePoint(event);
    shapeDraw.b = { col: point.col, row: point.row };
    const size = Math.hypot(shapeDraw.b.col - shapeDraw.a.col, shapeDraw.b.row - shapeDraw.a.row);
    shapeDraw.pending = size >= 2;
    if (shapeDraw.pending) showShapeConfirm();
    else shapeConfirm.hidden = true;
    draw();
    return;
  }
  const start = dragging;
  dragging = null;
  zoomBox = null;
  canvas.classList.remove("dragging");
  if (!start || !disp) return;
  const point = imagePoint(event);
  const moved = Math.hypot(point.x - start.x, point.y - start.y) > 4;
  if (start.mode !== "zoom" && !moved) {
    if (!pickObject(start.col, start.row)) clearPick();
  }
  if (start.mode === "select") return;
  if (start.mode === "zoom") {
    if (moved) zoomToRect(start.col, start.row, point.col, point.row);
    else zoomAtScreen(point.x, point.y, 1.4);
    draw();
  }
});
canvas.addEventListener("mousemove", (event) => {
  if (shapeDraw && shapeDraw.a && !shapeDraw.pending && event.buttons === 1) {
    const point = imagePoint(event);
    shapeDraw.b = { col: point.col, row: point.row };
    draw();
  }
  if (dragging && dragging.mode === "pan") {
    const point = imagePoint(event);
    const shifted = VisdevGeom.panBy(dragging.camX, dragging.camY, point.x - dragging.x, point.y - dragging.y);
    cam.x = shifted.x;
    cam.y = shifted.y;
    cam.user = true;
  } else if (dragging && dragging.mode === "zoom") {
    const point = imagePoint(event);
    zoomBox = { x: dragging.x, y: dragging.y, x2: point.x, y2: point.y };
  }
  updateProbe(event);
});
canvas.addEventListener("mouseleave", () => {
  mouse = null;
  hideProbe();
  stPix.textContent = "Row —   Column —   R —  G —  B —";
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

function vizionCall(name, args) {
  return `Vizion.${name} (${args})`;
}

function guessImageName(lines) {
  let name = "Image";
  for (const line of lines) {
    const match = line.match(/^\s*(?:Vizion\.)?(read_image|rgb1_to_gray|gauss_image|median_image|mean_image|emphasize|scale_image|invert_image)\s*\((.*)\)\s*$/i);
    if (!match) continue;
    const args = splitArgs(match[2]);
    const picked = match[1].toLowerCase() === "read_image" ? args[0] : args[1];
    if (picked && picked.trim()) name = picked.trim();
  }
  return name;
}

function applyBinaryTool(tool) {
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const index = lines.findIndex((line) => /^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\(/i.test(line));
  let image = guessImageName(lines);
  let region = "Region";
  let level = histo && Number.isFinite(histo.threshold) ? histo.threshold : (histo && Number.isFinite(histo.min) ? histo.min : 128);
  let maxGray = histo && histo.kind === "range" ? histo.max : 255;
  let used = "UsedThreshold";
  let indent = "";
  if (index >= 0) {
    const match = lines[index].match(/^(\s*)(?:Vizion\.)?(?:threshold|opencv_threshold|binary_threshold)\s*\((.*)\)\s*$/i);
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
  let call = vizionCall("binary_threshold", `${image}, ${region}, 'max_separability', 'light', ${used}`);
  if (tool === "halcon") call = vizionCall("threshold", `${image}, ${region}, ${level}, ${maxGray}`);
  else if (tool === "opencv") call = vizionCall("opencv_threshold", `${image}, ${region}, ${level}, 'binary'`);
  else if (tool === "inv") call = vizionCall("opencv_threshold", `${image}, ${region}, ${level}, 'binary_inv'`);
  else if (tool === "dark") call = vizionCall("binary_threshold", `${image}, ${region}, 'max_separability', 'dark', ${used}`);
  else if (tool === "smooth") call = vizionCall("binary_threshold", `${image}, ${region}, 'smooth_histo', 'light', ${used}`);
  else if (tool === "triangle") call = vizionCall("binary_threshold", `${image}, ${region}, 'triangle', 'light', ${used}`);
  else if (tool === "local") call = vizionCall("binary_threshold", `${image}, ${region}, 'local_mean', 'light', ${used}`);
  else if (tool === "lgauss") call = vizionCall("binary_threshold", `${image}, ${region}, 'local_gauss', 'light', ${used}`);
  if (index >= 0) lines[index] = indent + call;
  else {
    const displayAt = lines.findIndex((line) => /^\s*dev_display\s*\(/i.test(line));
    lines.splice(displayAt >= 0 ? displayAt : lines.length, 0, call);
  }
  const opencv = tool === "opencv" || tool === "inv";
  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    if (opencv) {
      const linked = line.match(/^(\s*)(?:Vizion\.)?connection\s*\(\s*([^,]+)\s*,\s*([^,)]+)\s*(?:,\s*([^)]+))?\s*\)\s*$/i);
      if (linked && linked[2].trim() === region) {
        const dist = linked[4] ? linked[4].trim() : "1";
        lines[lineIndex] = `${linked[1]}${vizionCall("opencv_connection", `${linked[2].trim()}, ${linked[3].trim()}, 8, ${dist}`)}`;
      }
    } else {
      const linked = line.match(/^(\s*)(?:Vizion\.)?opencv_connection\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*(?:,\s*([^)]+))?\s*\)\s*$/i);
      if (linked && linked[2].trim() === region) {
        const dist = linked[5] ? linked[5].trim() : "1";
        lines[lineIndex] = `${linked[1]}${vizionCall("connection", `${linked[2].trim()}, ${linked[3].trim()}, ${dist}`)}`;
      }
    }
  }
  code.value = lines.join("\n");
  commitProgram("checkpoint");
  renderEditor();
  exec("run");
}

function applyConnection(minArea, dist) {
  const area = Math.max(0, Math.round(Number(minArea) || 0));
  const distance = !Number.isFinite(Number(dist)) || Number(dist) < 1 ? 1 : Math.round(Number(dist) * 1000) / 1000;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  let region = "Region";
  let opencv = false;
  const bin = lines.findIndex((line) => /^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\(/i.test(line));
  if (bin >= 0) {
    const inner = lines[bin].match(/\((.*)\)/);
    const args = inner ? splitArgs(inner[1]).map((item) => item.trim()) : [];
    if (args[1]) region = args[1];
    opencv = /(?:Vizion\.)?opencv_threshold/i.test(lines[bin]);
  }
  const regionPattern = region.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const connPattern = new RegExp(`^\\s*(?:Vizion\\.)?(connection|opencv_connection)\\s*\\(\\s*${regionPattern}\\s*,`, "i");
  let connIndex = lines.findIndex((line) => connPattern.test(line));
  let output = "Connected";
  if (connIndex >= 0) {
    const match = lines[connIndex].match(/^(\s*)(?:Vizion\.)?(connection|opencv_connection)\s*\((.*)\)\s*$/i);
    const args = splitArgs(match[3]).map((item) => item.trim());
    output = args[1] || output;
    if (match[2].toLowerCase() === "opencv_connection") {
      const linked = args[2] || "8";
      lines[connIndex] = `${match[1]}${vizionCall("opencv_connection", `${args[0]}, ${output}, ${linked}, ${distance}`)}`;
    } else {
      lines[connIndex] = `${match[1]}${vizionCall("connection", `${args[0]}, ${output}, ${distance}`)}`;
    }
  } else {
    const insertAt = bin >= 0 ? bin + 1 : lines.length;
    const call = opencv
      ? vizionCall("opencv_connection", `${region}, ${output}, 8, ${distance}`)
      : vizionCall("connection", `${region}, ${output}, ${distance}`);
    lines.splice(insertAt, 0, call);
    connIndex = insertAt;
  }
  const outputPattern = output.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const shapePattern = new RegExp(`^\\s*(?:Vizion\\.)?select_shape\\s*\\(\\s*${outputPattern}\\s*,`, "i");
  const shapeIndex = lines.findIndex((line, index) => index > connIndex && shapePattern.test(line) && /area/i.test(line));
  let shown = output;
  if (shapeIndex >= 0) {
    const match = lines[shapeIndex].match(/^(\s*)(?:Vizion\.)?select_shape\s*\((.*)\)\s*$/i);
    const args = splitArgs(match[2]);
    if (args.length >= 6) {
      args[4] = ` ${area}`;
      lines[shapeIndex] = `${match[1]}${vizionCall("select_shape", args.map((item) => item.trim()).join(", "))}`;
    }
    if (args[1]) shown = args[1].trim();
  } else {
    shown = "Selected";
    lines.splice(connIndex + 1, 0, vizionCall("select_shape", `${output}, Selected, 'area', 'and', ${area}, 99999999`));
  }
  const shownPattern = shown.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const alreadyShown = lines.some((line) => new RegExp(`^\\s*(?:Vizion\\.)?dev_display\\s*\\(\\s*${shownPattern}\\s*\\)`, "i").test(line));
  if (!alreadyShown) {
    const regionDisplay = new RegExp(`^(\\s*)(?:Vizion\\.)?dev_display\\s*\\(\\s*${regionPattern}\\s*\\)\\s*$`, "i");
    const displayIndex = lines.findIndex((line) => regionDisplay.test(line));
    if (displayIndex >= 0) {
      lines[displayIndex] = `${lines[displayIndex].match(/^\s*/)[0]}${vizionCall("dev_display", shown)}`;
    } else {
      lines.push(vizionCall("dev_set_draw", "'fill'"));
      lines.push(vizionCall("dev_display", shown));
    }
  }
  code.value = lines.join("\n");
  commitProgram("checkpoint");
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
  const match = original.match(/^(\s*)(?:Vizion\.)?threshold_sub_pix\s*\((.*)\)\s*$/i);
  if (!match) return false;
  const args = splitArgs(match[2]);
  if (args.length < 3 || !/^-?\d+(\.\d+)?$/.test(args[2].trim())) return false;
  args[2] = ` ${level}`;
  lines[histo.line - 1] = `${match[1]}${vizionCall("threshold_sub_pix", args.join(","))}`;
  code.value = lines.join("\n");
  commitProgram("checkpoint");
  renderEditor();
  return true;
}

function applySubpix(level) {
  const value = Math.max(0, Math.min(255, Math.round(Number(level) || 0)));
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const image = guessImageName(lines);
  let regionName = "Region";
  const bin = lines.findIndex((line) => /^\s*(?:Vizion\.)?(threshold|opencv_threshold|binary_threshold)\s*\(/i.test(line));
  if (bin >= 0) {
    const inner = lines[bin].match(/\((.*)\)/);
    const args = inner ? splitArgs(inner[1]).map((item) => item.trim()) : [];
    if (args[1]) regionName = args[1];
  } else {
    const at = lines.findIndex((line) => /^\s*(?:Vizion\.)?dev_display\s*\(/i.test(line));
    lines.splice(at >= 0 ? at : lines.length, 0, vizionCall("threshold", `${image}, ${regionName}, ${value}, 255`));
  }
  let output = "Border";
  const index = lines.findIndex((line) => /^\s*(?:Vizion\.)?threshold_sub_pix\s*\(/i.test(line));
  if (index >= 0) {
    const match = lines[index].match(/^(\s*)(?:Vizion\.)?threshold_sub_pix\s*\((.*)\)\s*$/i);
    const args = splitArgs(match[2]).map((item) => item.trim());
    output = args[1] || output;
    lines[index] = `${match[1]}${vizionCall("threshold_sub_pix", `${args[0] || image}, ${output}, ${value}`)}`;
  } else {
    const at = lines.findIndex((line) => /^\s*(?:Vizion\.)?dev_display\s*\(/i.test(line));
    lines.splice(at >= 0 ? at : lines.length, 0, vizionCall("threshold_sub_pix", `${image}, ${output}, ${value}`));
  }
  const shown = (name) => lines.some((line) => new RegExp(`^\\s*(?:Vizion\\.)?dev_display\\s*\\(\\s*${name}\\s*\\)`, "i").test(line));
  if (!shown(regionName)) lines.push(vizionCall("dev_display", regionName));
  if (!shown(output)) lines.push(vizionCall("dev_display", output));
  code.value = lines.join("\n");
  commitProgram("checkpoint");
  renderEditor();
  exec("run");
}

function subpixNames() {
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  let region = "Region";
  let contour = "Border";
  for (const line of lines) {
    const binary = line.match(/^\s*(?:Vizion\.)?(?:threshold|opencv_threshold|binary_threshold)\s*\(\s*[^,]+,\s*([^,)]+)/i);
    if (binary) region = binary[1].trim();
    const sub = line.match(/^\s*(?:Vizion\.)?threshold_sub_pix\s*\(\s*[^,]+,\s*([^,)]+)/i);
    if (sub) contour = sub[1].trim();
  }
  return { region, contour };
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
  const names = subpixNames();
  const url = `${location.origin}/bin?id=${encodeURIComponent(sessionId || "")}&level=${encodeURIComponent(level)}&region=${encodeURIComponent(names.region)}&contour=${encodeURIComponent(names.contour)}`;
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_bin_window) {
      await window.pywebview.api.open_bin_window(url);
      return;
    }
  } catch (_err) {
    /* 沒有桌面視窗介面時改開瀏覽器視窗 */
  }
  const popup = window.open(url, "visdev-subpix", "popup=yes,width=920,height=720,left=80,top=40");
  if (!popup) applySubpix(level);
}

function rewriteOpenCV(level) {
  if (!histo || !histo.literal || histo.kind !== "opencv" || !histo.line) return false;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const original = lines[histo.line - 1];
  if (!original) return false;
  const match = original.match(/^(\s*)(?:Vizion\.)?opencv_threshold\s*\((.*)\)\s*$/i);
  if (!match) return false;
  const args = splitArgs(match[2]);
  if (args.length < 3 || !/^-?\d+(\.\d+)?$/.test(args[2].trim())) return false;
  args[2] = ` ${level}`;
  lines[histo.line - 1] = `${match[1]}${vizionCall("opencv_threshold", args.join(","))}`;
  code.value = lines.join("\n");
  commitProgram("checkpoint");
  renderEditor();
  return true;
}

function rewriteThreshold(minGray, maxGray) {
  if (!histo || !histo.literal || histo.kind !== "range" || !histo.line) return false;
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const index = histo.line - 1;
  const original = lines[index];
  if (!original) return false;
  const match = original.match(/^(\s*)(?:Vizion\.)?threshold\s*\((.*)\)\s*$/i);
  if (!match) return false;
  const args = splitArgs(match[2]);
  if (args.length < 2) return false;
  if (!/^-?\d+(\.\d+)?$/.test(args.at(-1).trim()) || !/^-?\d+(\.\d+)?$/.test(args.at(-2).trim())) return false;
  args[args.length - 2] = ` ${minGray}`;
  args[args.length - 1] = ` ${maxGray}`;
  lines[index] = `${match[1]}${vizionCall("threshold", args.join(","))}`;
  code.value = lines.join("\n");
  commitProgram("checkpoint");
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
    imageStamp,
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
function applyOcr() {
  const method = document.getElementById("ocrMethod").value || "auto";
  const lines = code.value.replace(/\r\n/g, "\n").split("\n");
  const image = guessImageName(lines);
  const call = vizionCall("ocr_find", `${image}, Words, '${method}', Text, Row, Column, Score`);
  const index = lines.findIndex((line) => /^\s*(?:Vizion\.)?ocr_find\s*\(/i.test(line));
  if (index >= 0) lines[index] = call;
  else lines.push(call);
  code.value = lines.join("\n");
  commitProgram("checkpoint");
  renderEditor();
  exec("run");
}

document.getElementById("btnThreshold").addEventListener("click", () => openThresholdWindow());
document.getElementById("btnFeatures").addEventListener("click", () => openFeatureWindow());
document.getElementById("btnOcr").addEventListener("click", () => applyOcr());
document.getElementById("btnConnection").addEventListener("click", () => {
  applyConnection(document.getElementById("connMin").value, document.getElementById("connDist").value);
});
setInterval(() => {
  const raw = localStorage.getItem("visdev.thresholdCmd");
  if (!raw) return;
  try { handleThresholdCmd(JSON.parse(raw)); } catch (_err) { /* 忽略壞掉的暫存 */ }
}, 300);

function sampleButtons(items) {
  return (items || []).map((item) => {
    const title = item.title || item.name;
    return `<button type="button" data-sample="${esc(item.name)}" data-kind="${esc(item.kind)}" data-name="${esc(item.name)}"><b>${esc(title)}</b><span>${esc(item.name)}</span></button>`;
  }).join("");
}

async function openProgram(kind, name) {
  const res = await fetch(`/api/files/${kind}/${encodeURIComponent(name)}`);
  const data = await res.json();
  if (!res.ok) {
    showDetail(data);
    return;
  }
  code.value = data.source.replace(/\r\n/g, "\n");
  document.getElementById("progName").value = data.name;
  saveTarget = { kind, name: data.name };
  sessionId = null;
  breakpoints = new Set();
  sampleMenu.hidden = true;
  resetProgramHistory();
  markSaved();
  renderEditor();
  await exec("reset");
}

async function loadFiles() {
  const files = await fetch("/api/files").then((res) => res.json());
  const block = (title, items) => {
    if (!items.length) return "";
    return `<div class="grouplab">${esc(title)}</div>` + sampleButtons(items);
  };
  sampleMenu.innerHTML = block("範例", files.samples || []) + block("已儲存", files.workspace || []);
  const fly = document.getElementById("sampleFly");
  if (fly) fly.innerHTML = sampleButtons(files.samples || []) || `<div class="hint" style="padding:8px">沒有範例</div>`;
}

sampleMenu.addEventListener("click", (event) => {
  const node = event.target.closest("button");
  if (!node || !node.dataset.name) return;
  openProgram(node.dataset.kind || "samples", node.dataset.name);
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

let displayChannel = null;

function currentFrame() {
  if (!cam.user || !disp || !disp.width) return null;
  const rect = canvas.getBoundingClientRect();
  const center = VisdevGeom.imagePoint(rect.width / 2, rect.height / 2, cam);
  return { scale: cam.scale, col: center.col, row: center.row };
}

function publishDisplay(display) {
  const copy = display ? {
    width: display.width,
    height: display.height,
    image: display.image || null,
    shapes: (display.shapes || []).map((shape) => {
      const next = { ...shape };
      delete next._img;
      return next;
    }),
  } : null;
  const payload = { type: "display", id: sessionId, display: copy, displayRev: liveDisplayRev, frame: currentFrame(), at: Date.now() };
  try {
    if (!displayChannel) displayChannel = new BroadcastChannel("visdev-display");
    displayChannel.postMessage(payload);
  } catch (_err) { /* 拉出的視窗會自己讀目前畫面 */ }
}

async function openGfxWindow() {
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  if (!sessionId) await exec("run");
  while (inflight) await new Promise((resolve) => setTimeout(resolve, 100));
  const url = `${location.origin}/gfx?id=${encodeURIComponent(sessionId || "")}`;
  try {
    if (window.pywebview && window.pywebview.api && window.pywebview.api.open_gfx_window) {
      await window.pywebview.api.open_gfx_window(url);
      return;
    }
  } catch (_err) { /* 沒有桌面視窗介面時改開瀏覽器視窗 */ }
  window.open(url, "visdev-gfx", "popup=yes,width=960,height=720,left=120,top=60");
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
  window.open(url, "visdev-features", "popup=yes,width=880,height=720,left=420,top=60");
}

function showPickedVariable(name) {
  if (!name || name === "__preview__" || selectedVar === name) return;
  selectedVar = name;
  renderVars();
  const node = [...document.querySelectorAll("#iconicList .var")].find((item) => item.dataset.name === name);
  if (node) node.scrollIntoView({ block: "nearest" });
}

function pickBanner(pick) {
  const label = pick.kind === "xld" ? "輪廓" : "區域";
  const shown = pick.name === "__preview__" ? "次像素" : (pick.name || "");
  const body = `已點選${label} ${shown} #${pick.index + 1}。白色虛線是這一個，特徵只顯示它。`;
  return pick.summary ? `${body} ${pick.summary}` : body;
}

let pickTicket = 0;

async function describePick() {
  const pick = pickedObject;
  if (!pick || !sessionId || pick.name === "__preview__") return;
  const ticket = ++pickTicket;
  let res;
  try {
    res = await fetch("/api/features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        id: sessionId,
        name: pick.name,
        index: pick.index,
        kind: pick.kind || "region",
        region: ["area", "row", "column"],
        xld: ["area", "row", "column"],
        gray: [],
        subpixel: false,
      }),
    });
  } catch (_err) {
    return;
  }
  const data = await res.json().catch(() => null);
  if (ticket !== pickTicket || pickedObject !== pick || !res.ok || !data) return;
  const values = (pick.kind === "xld" ? data.xld : data.region) || {};
  const text = VisdevPick.summary(values);
  if (!text) return;
  pick.summary = text;
  gfxBanner.textContent = pickBanner(pick);
  publishFeaturePick();
}

function publishFeaturePick() {
  if (!pickedObject) return;
  const payload = VisdevPick.pack("main", pickedObject, sessionId);
  localStorage.setItem("visdev.featurePick", JSON.stringify(payload));
  featureChannel.postMessage(payload);
}

function clearPick() {
  pickedObject = null;
  featureLines = [];
  gfxBanner.hidden = true;
  const payload = VisdevPick.pack("main", null, sessionId);
  localStorage.setItem("visdev.featurePick", JSON.stringify(payload));
  featureChannel.postMessage(payload);
  draw();
}

function pickObject(col, row) {
  const hit = VisdevGeom.pickShape(disp?.shapes || [], col, row, cam.scale);
  if (!hit) return false;
  pickedObject = hit;
  showPickedVariable(pickedObject.name);
  gfxBanner.hidden = false;
  gfxBanner.textContent = pickBanner(pickedObject);
  draw();
  publishFeaturePick();
  loadFeatures();
  describePick();
  return true;
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
const pickWatch = VisdevPick.createWatch("main");
function applyExternalPick(data) {
  const next = pickWatch.accept(data);
  if (!next || VisdevPick.sameSelection(pickedObject, next.selection)) return;
  if (!next.selection) {
    pickedObject = null;
    featureLines = [];
    gfxBanner.hidden = true;
    draw();
    return;
  }
  pickedObject = next.selection;
  showPickedVariable(pickedObject.name);
  gfxBanner.hidden = false;
  gfxBanner.textContent = pickBanner(pickedObject);
  draw();
  loadFeatures();
  if (!pickedObject.summary) describePick();
}

function readStoredPick() {
  try {
    return JSON.parse(localStorage.getItem("visdev.featurePick") || "null");
  } catch (_err) {
    return null;
  }
}

featureChannel.onmessage = (event) => {
  const data = event.data;
  if (!data) return;
  if (data.type === "features" && pickedObject) loadFeatures();
  applyExternalPick(data);
};
window.addEventListener("storage", (event) => {
  if (event.key === "visdev.features" && pickedObject) loadFeatures();
  if (event.key === "visdev.featurePick" && event.newValue) {
    try { applyExternalPick(JSON.parse(event.newValue)); } catch (_err) { /* 忽略壞掉的點選 */ }
  }
});
setInterval(() => applyExternalPick(readStoredPick()), 300);

function fillOcrMethods(engines) {
  const select = document.getElementById("ocrMethod");
  if (!select || !engines) return;
  const current = select.value || "auto";
  const groups = [];
  engines.forEach((item) => {
    if (!groups.includes(item.group)) groups.push(item.group);
  });
  select.innerHTML = groups.map((group) => {
    const options = engines.filter((item) => item.group === group).map((item) => {
      const mark = item.key === "auto" || item.installed ? "" : "（未安裝）";
      return `<option value="${esc(item.key)}" title="${esc(item.purpose)}">${esc(item.title)}${mark}</option>`;
    }).join("");
    return `<optgroup label="${esc(group)}">${options}</optgroup>`;
  }).join("");
  if ([...select.options].some((option) => option.value === current)) select.value = current;
}

async function boot() {
  operators = await fetch("/api/operators").then((res) => res.json());
  try {
    fillOcrMethods(await fetch("/api/ocr").then((res) => res.json()));
  } catch (_err) {
    /* 目錄讀不到時，下拉仍保留自動 */
  }
  operatorNames = new Set(operators.map((op) => op.name.toLowerCase()));
  buildOperatorMenu();
  renderOps();
  if (operators[0]) showOp(operators[0].name);
  await loadFiles();
  const sample = await fetch("/api/files/samples/blob_inspect.hdv").then((res) => res.json());
  code.value = sample.source;
  document.getElementById("progName").value = sample.name;
  saveTarget = { kind: "sample", name: sample.name };
  resetProgramHistory();
  markSaved();
  renderEditor();
  resizeCanvas();
  await exec("reset");
}

boot();
