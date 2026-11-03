(function () {
  function signature(item, kind) {
    if (kind === "control") {
      return JSON.stringify([item.name, item.scope || "", item.type || "", item.text || "", item.full || ""]);
    }
    return JSON.stringify([
      item.name,
      item.scope || "",
      item.role || "",
      item.kind || "",
      item.info || "",
      item.thumb || "",
    ]);
  }

  function iconicMeta(item) {
    const scope = item.scope ? item.scope + " · " : "";
    const role = item.role && item.role !== item.kind ? item.role + " · " + item.kind : (item.kind || "");
    return scope + role + " · " + (item.info || "");
  }

  function controlSmall(item) {
    return (item.scope ? item.scope + " " : "") + (item.type || "");
  }

  function createButton(item, kind, selected) {
    const button = document.createElement("button");
    button.className = (kind === "control" ? "var ctrl" : "var") + (selected ? " on" : "");
    button.dataset.name = item.name;
    button.dataset.kind = kind;
    button._visdevSig = signature(item, kind);
    if (kind === "control") {
      button.title = item.full || item.text || "";
      const name = document.createElement("span");
      name.className = "vn";
      name.appendChild(document.createTextNode(item.name));
      const small = document.createElement("small");
      small.textContent = controlSmall(item);
      name.appendChild(small);
      const value = document.createElement("span");
      value.className = "vm";
      value.textContent = item.text || "";
      button.appendChild(name);
      button.appendChild(value);
    } else {
      const img = document.createElement("img");
      img.alt = "";
      img.src = item.thumb || "";
      const name = document.createElement("span");
      name.className = "vn";
      name.textContent = item.name;
      const meta = document.createElement("span");
      meta.className = "vm";
      meta.textContent = iconicMeta(item);
      button.appendChild(img);
      button.appendChild(name);
      button.appendChild(meta);
    }
    return button;
  }

  function updateButton(button, item, kind, selected) {
    button.classList.toggle("on", !!selected);
    button._visdevSig = signature(item, kind);
    if (kind === "control") {
      const title = item.full || item.text || "";
      if (button.title !== title) button.title = title;
      const name = button.querySelector(".vn");
      const text = name.firstChild;
      if (text && text.nodeValue !== item.name) text.nodeValue = item.name;
      const small = name.querySelector("small");
      const smallText = controlSmall(item);
      if (small && small.textContent !== smallText) small.textContent = smallText;
      const value = button.querySelector(".vm");
      const shown = item.text || "";
      if (value.textContent !== shown) value.textContent = shown;
      return;
    }
    const img = button.querySelector("img");
    const thumb = item.thumb || "";
    if (img.getAttribute("src") !== thumb) img.setAttribute("src", thumb);
    const name = button.querySelector(".vn");
    if (name.textContent !== item.name) name.textContent = item.name;
    const meta = button.querySelector(".vm");
    const metaText = iconicMeta(item);
    if (meta.textContent !== metaText) meta.textContent = metaText;
  }

  function rowButtons(list) {
    return [...list.children].filter((node) => node.classList && node.classList.contains("var"));
  }

  function syncVarList(list, items, kind, selected, emptyText) {
    const rows = items || [];
    const stats = { created: 0, updated: 0, kept: 0, removed: 0 };
    if (!rows.length) {
      const hint = list.children.length === 1 && list.children[0].classList.contains("hint") ? list.children[0] : null;
      if (hint && hint.textContent === emptyText) return stats;
      while (list.firstChild) list.removeChild(list.firstChild);
      const node = document.createElement("div");
      node.className = "hint";
      node.textContent = emptyText;
      list.appendChild(node);
      stats.created = 1;
      return stats;
    }
    [...list.children].forEach((node) => {
      if (node.classList.contains("hint")) node.remove();
    });
    const byName = new Map(rowButtons(list).map((node) => [node.dataset.name, node]));
    const seen = new Set();
    rows.forEach((item, index) => {
      const sig = signature(item, kind);
      const on = selected === item.name;
      let node = byName.get(item.name);
      if (!node || node.dataset.kind !== kind) {
        node = createButton(item, kind, on);
        stats.created += 1;
      } else if (node._visdevSig !== sig) {
        updateButton(node, item, kind, on);
        stats.updated += 1;
      } else if (node.classList.contains("on") !== !!on) {
        node.classList.toggle("on", !!on);
        stats.kept += 1;
      } else {
        stats.kept += 1;
      }
      seen.add(item.name);
      const at = list.children[index];
      if (at !== node) list.insertBefore(node, at || null);
    });
    rowButtons(list).forEach((node) => {
      if (!seen.has(node.dataset.name)) {
        node.remove();
        stats.removed += 1;
      }
    });
    return stats;
  }

  function syncLog(box, lines) {
    const wanted = (lines || []).map((line) => String(line));
    const spans = () => [...box.children].filter((node) => String(node.tagName || "").toUpperCase() === "SPAN");
    const have = spans();
    let keep = 0;
    while (keep < have.length && keep < wanted.length) {
      const bad = wanted[keep].startsWith("錯誤");
      if (have[keep].textContent !== wanted[keep] || have[keep].classList.contains("bad") !== bad) break;
      keep += 1;
    }
    for (let index = have.length - 1; index >= keep; index -= 1) {
      const span = have[index];
      const prev = span.previousSibling;
      if (prev && prev.nodeType === 3 && prev.nodeValue === "\n") prev.remove();
      span.remove();
    }
    for (let index = keep; index < wanted.length; index += 1) {
      if (spans().length) box.appendChild(document.createTextNode("\n"));
      const span = document.createElement("span");
      if (wanted[index].startsWith("錯誤")) span.className = "bad";
      span.textContent = wanted[index];
      box.appendChild(span);
    }
    return { kept: keep, appended: wanted.length - keep };
  }

  function problemBase(item) {
    return JSON.stringify([item.sev, item.code || "", item.text || "", item.file || "", item.line || ""]);
  }

  function makeCell(className, text) {
    const cell = document.createElement("td");
    if (className) cell.className = className;
    cell.textContent = text == null ? "" : String(text);
    return cell;
  }

  function createProblemRow(item, key, labelOf) {
    const row = document.createElement("tr");
    row.dataset.line = item.line == null || item.line === "" ? "" : String(item.line);
    row._visdevKey = key;
    row.appendChild(makeCell("sev-" + item.sev, labelOf[item.sev] || item.sev));
    row.appendChild(makeCell("", item.code || ""));
    row.appendChild(makeCell("desc", item.text || ""));
    row.appendChild(makeCell("", item.file || ""));
    row.appendChild(makeCell("", item.line == null || item.line === "" ? "" : String(item.line)));
    return row;
  }

  function emptyProblemRow() {
    const row = document.createElement("tr");
    row.dataset.empty = "1";
    row._visdevKey = "";
    row.appendChild(makeCell("", ""));
    row.appendChild(makeCell("", ""));
    row.appendChild(makeCell("desc", "沒有要顯示的項目"));
    row.appendChild(makeCell("", ""));
    row.appendChild(makeCell("", ""));
    return row;
  }

  function syncProblems(body, items, labelOf) {
    const labels = labelOf || { error: "錯誤", warning: "警告", message: "訊息" };
    const rows = items || [];
    const stats = { created: 0, updated: 0, kept: 0, removed: 0 };
    if (!rows.length) {
      const hint = body.children.length === 1 && body.children[0].dataset.empty === "1" ? body.children[0] : null;
      if (hint) return stats;
      while (body.firstChild) body.removeChild(body.firstChild);
      body.appendChild(emptyProblemRow());
      stats.created = 1;
      return stats;
    }
    [...body.children].forEach((node) => {
      if (node.dataset && node.dataset.empty === "1") node.remove();
    });
    const seen = new Map();
    const keyed = rows.map((item) => {
      const base = problemBase(item);
      const count = (seen.get(base) || 0) + 1;
      seen.set(base, count);
      return { item: item, key: base + "#" + count };
    });
    const byKey = new Map([...body.children].filter((node) => node._visdevKey).map((node) => [node._visdevKey, node]));
    const used = new Set();
    keyed.forEach((entry, index) => {
      let node = byKey.get(entry.key);
      if (!node) {
        node = createProblemRow(entry.item, entry.key, labels);
        stats.created += 1;
      } else {
        stats.kept += 1;
      }
      used.add(entry.key);
      const at = body.children[index];
      if (at !== node) body.insertBefore(node, at || null);
    });
    [...body.children].forEach((node) => {
      if (!used.has(node._visdevKey)) {
        node.remove();
        stats.removed += 1;
      }
    });
    return stats;
  }

  const api = {
    signature: signature,
    syncVarList: syncVarList,
    iconicMeta: iconicMeta,
    syncLog: syncLog,
    syncProblems: syncProblems,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevVars = api;
})();
