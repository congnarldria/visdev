(function () {
  const LABELS = { Dia: "直徑", Width: "寬", Height: "高", Dark: "暗塊", Bright: "亮點" };

  function readLimits(source) {
    const lines = String(source || "").split("\n");
    const found = new Map();
    lines.forEach((line, index) => {
      const match = line.match(/^(\s*)([A-Za-z_]\w*?)(Min|Max)\s*:=\s*(-?\d+(?:\.\d+)?)\s*$/);
      if (!match) return;
      const stem = match[2];
      const slot = found.get(stem) || {
        stem: stem,
        label: LABELS[stem] || stem,
        min: null,
        max: null,
        minLine: null,
        maxLine: null,
        minName: stem + "Min",
        maxName: stem + "Max",
      };
      if (match[3] === "Min") {
        slot.min = match[4];
        slot.minLine = index;
      } else {
        slot.max = match[4];
        slot.maxLine = index;
      }
      found.set(stem, slot);
    });
    return [...found.values()]
      .filter((item) => item.min != null && item.max != null)
      .sort((a, b) => a.minLine - b.minLine);
  }

  function applyLimit(source, lineIndex, name, value) {
    const lines = String(source || "").split("\n");
    const index = Number(lineIndex);
    if (!Number.isInteger(index) || index < 0 || index >= lines.length) return String(source || "");
    if (!/^-?\d+(?:\.\d+)?$/.test(String(value))) return String(source || "");
    const indent = (lines[index].match(/^\s*/) || [""])[0];
    lines[index] = indent + name + " := " + value;
    return lines.join("\n");
  }

  const api = { readLimits: readLimits, applyLimit: applyLimit };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevTol = api;
})();
