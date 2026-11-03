(function () {
  function pack(from, hit, id) {
    const message = {
      type: "pick",
      from: from || "",
      id: id || "",
      at: Date.now(),
    };
    if (!hit || !hit.name) {
      message.clear = true;
      return message;
    }
    message.name = hit.name;
    message.index = hit.index;
    message.kind = hit.kind || "region";
    if (hit.group != null) message.group = hit.group;
    if (hit.summary) message.summary = String(hit.summary);
    return message;
  }

  function summary(values) {
    if (!values || values.area == null || values.area === "") return "";
    const area = values.area;
    const row = Number(values.row);
    const column = Number(values.column);
    if (!Number.isFinite(row) || !Number.isFinite(column)) return `面積 ${area}`;
    return `面積 ${area}  中心 ${row.toFixed(1)}, ${column.toFixed(1)}`;
  }

  function graphicsSelection(message) {
    if (!message || message.type !== "pick" || message.clear || !message.name || message.name === "__preview__") return null;
    const kind = message.kind || "region";
    const selection = {
      name: message.name,
      index: message.index,
      kind: kind,
      group: kind === "xld" ? (message.group == null ? message.index : message.group) : null,
    };
    if (message.summary) selection.summary = String(message.summary);
    return selection;
  }

  function sameSelection(left, right) {
    if (!left && !right) return true;
    if (!left || !right) return false;
    return left.name === right.name && left.index === right.index && left.kind === right.kind;
  }

  function createWatch(ownFrom) {
    let seen = 0;
    function accept(message, force) {
      if (!message || message.type !== "pick") return null;
      const at = message.at || 0;
      if (at && at === seen && !force) return null;
      if (!force && message.from === ownFrom) {
        if (at) seen = at;
        return null;
      }
      if (at) seen = at;
      return { selection: graphicsSelection(message) };
    }
    return { accept: accept };
  }

  const api = {
    pack: pack,
    summary: summary,
    graphicsSelection: graphicsSelection,
    sameSelection: sameSelection,
    createWatch: createWatch,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevPick = api;
})();
