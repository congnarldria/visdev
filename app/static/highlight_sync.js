(function () {
  function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
  }

  function highlightSource(line, isOperator) {
    if (/^\s*(?:\*|\/\/)/.test(line)) return `<span class="tok-cmt">${escapeHtml(line)}</span>`;
    const re = /('[^']*'|"[^"]*"|[A-Za-z_]\w*\.[A-Za-z_]\w*|\b(?:if|else|endif|for|endfor|while|endwhile)\b|\b\d+(?:\.\d+)?\b|[A-Za-z_]\w*|\s+|.)/g;
    let html = "";
    let match;
    const known = typeof isOperator === "function" ? isOperator : () => false;
    while ((match = re.exec(line))) {
      const token = match[0];
      let cls = "";
      if (token[0] === "'" || token[0] === '"') cls = "tok-str";
      else if (/^\d/.test(token)) cls = "tok-num";
      else if (/^(if|else|endif|for|endfor|while|endwhile)$/i.test(token)) cls = "tok-key";
      else if (known(token)) cls = "tok-op";
      html += cls ? `<span class="${cls}">${escapeHtml(token)}</span>` : escapeHtml(token);
    }
    return html || " ";
  }

  function normalizeNewlines(container) {
    const lines = [...container.children];
    [...container.childNodes].forEach((node) => {
      if (node.nodeType === 3) node.remove();
    });
    for (let index = 1; index < lines.length; index += 1) {
      container.insertBefore(document.createTextNode("\n"), lines[index]);
    }
  }

  function syncHighlight(container, lines, renderHtml) {
    const next = lines || [];
    let nodes = [...container.children].filter((node) => node.classList && node.classList.contains("hl-line"));
    if (!container._visdevLines || container._visdevLines.length !== nodes.length) {
      while (container.firstChild) container.removeChild(container.firstChild);
      nodes = [];
      container._visdevLines = [];
    }
    const before = container._visdevLines;
    let start = 0;
    while (start < before.length && start < next.length && before[start] === next[start]) start += 1;
    let prevEnd = before.length - 1;
    let nextEnd = next.length - 1;
    while (prevEnd >= start && nextEnd >= start && before[prevEnd] === next[nextEnd]) {
      prevEnd -= 1;
      nextEnd -= 1;
    }
    const stats = {
      kept: start + Math.max(0, before.length - (prevEnd + 1)),
      updated: 0,
      created: 0,
      removed: 0,
    };
    const middleOld = prevEnd >= start ? prevEnd - start + 1 : 0;
    const middleNew = nextEnd >= start ? nextEnd - start + 1 : 0;
    if (middleOld === middleNew && middleOld > 0) {
      for (let index = start; index <= prevEnd; index += 1) {
        nodes[index].innerHTML = renderHtml(next[index]);
        stats.updated += 1;
      }
    } else if (middleOld > 0 || middleNew > 0) {
      const anchor = nodes[prevEnd + 1] || null;
      for (let index = prevEnd; index >= start; index -= 1) {
        nodes[index].remove();
        stats.removed += 1;
      }
      for (let index = start; index <= nextEnd; index += 1) {
        const line = document.createElement("span");
        line.className = "hl-line";
        line.innerHTML = renderHtml(next[index]);
        container.insertBefore(line, anchor);
        stats.created += 1;
      }
      normalizeNewlines(container);
    }
    container._visdevLines = next.slice();
    return stats;
  }

  function syncRows(container, rows) {
    const wanted = rows || [];
    const stats = { kept: 0, updated: 0, created: 0, removed: 0 };
    const nodes = [...container.children];
    while (nodes.length > wanted.length) {
      nodes.pop().remove();
      stats.removed += 1;
    }
    wanted.forEach((row, index) => {
      const className = row.className || "";
      const html = row.html == null ? null : String(row.html);
      const text = row.text == null ? null : String(row.text);
      let node = nodes[index];
      if (!node) {
        node = document.createElement("div");
        node.className = className;
        if (row.line != null) node.dataset.line = String(row.line);
        if (html != null) {
          node.innerHTML = html;
          node._visdevHtml = html;
        } else {
          node.textContent = text || "";
        }
        node._visdevSig = className + "\n" + (html != null ? html : text || "");
        container.appendChild(node);
        stats.created += 1;
        return;
      }
      const sig = className + "\n" + (html != null ? html : text || "");
      if (node._visdevSig === sig) {
        stats.kept += 1;
        return;
      }
      if (node.className !== className) node.className = className;
      if (html != null) {
        if (node._visdevHtml !== html) {
          node.innerHTML = html;
          node._visdevHtml = html;
        }
      } else if (node.textContent !== (text || "")) {
        node.textContent = text || "";
        node._visdevHtml = null;
      }
      if (row.line != null) node.dataset.line = String(row.line);
      node._visdevSig = sig;
      stats.updated += 1;
    });
    return stats;
  }

  const api = { syncHighlight: syncHighlight, highlightSource: highlightSource, syncRows: syncRows };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevHighlight = api;
})();
