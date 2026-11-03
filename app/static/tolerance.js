(function () {
  const LABELS = { Dia: "直徑", Width: "寬", Height: "高", Dark: "暗塊", Bright: "亮點", Hue: "色調", Area: "面積" };

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

  function readExpects(source) {
    const lines = String(source || "").split("\n");
    const found = [];
    lines.forEach((line, index) => {
      const match = line.match(/^(\s*)(Expect)\s*:=\s*'([^']*)'\s*$/);
      if (!match) return;
      found.push({
        name: match[2],
        label: "預期字",
        value: match[3],
        line: index,
      });
    });
    return found;
  }

  function applyExpect(source, lineIndex, name, value) {
    const lines = String(source || "").split("\n");
    const index = Number(lineIndex);
    const shown = String(value == null ? "" : value);
    if (!Number.isInteger(index) || index < 0 || index >= lines.length) return String(source || "");
    if (name !== "Expect" || /['\r\n\\]/.test(shown)) return String(source || "");
    const indent = (lines[index].match(/^\s*/) || [""])[0];
    lines[index] = indent + "Expect := '" + shown + "'";
    return lines.join("\n");
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

  const STORE_KEY = "visdev.tolerances";

  function fileKey(name) {
    return String(name || "").replace(/\\/g, "/").split("/").pop();
  }

  function readStore(storage) {
    if (!storage) return {};
    try {
      const parsed = JSON.parse(storage.getItem(STORE_KEY) || "{}");
      return parsed && typeof parsed === "object" ? parsed : {};
    } catch (_err) {
      return {};
    }
  }

  function writeStore(storage, data) {
    storage.setItem(STORE_KEY, JSON.stringify(data));
  }

  function currentValues(source) {
    const values = {};
    readLimits(source).forEach((field) => {
      values[field.minName] = field.min;
      values[field.maxName] = field.max;
    });
    readExpects(source).forEach((field) => {
      values[field.name] = field.value;
    });
    return values;
  }

  function remember(storage, fileName, source) {
    const key = fileKey(fileName);
    if (!key || !storage) return;
    const values = currentValues(source);
    if (!Object.keys(values).length) return;
    const all = readStore(storage);
    all[key] = values;
    writeStore(storage, all);
  }

  function restore(storage, fileName, source) {
    const text = String(source || "");
    const key = fileKey(fileName);
    const saved = key && storage ? readStore(storage)[key] : null;
    if (!saved) return text;
    let next = text;
    readLimits(next).forEach((field) => {
      if (saved[field.minName] != null) next = applyLimit(next, field.minLine, field.minName, saved[field.minName]);
      if (saved[field.maxName] != null) next = applyLimit(next, field.maxLine, field.maxName, saved[field.maxName]);
    });
    readExpects(next).forEach((field) => {
      if (saved[field.name] != null) next = applyExpect(next, field.line, field.name, saved[field.name]);
    });
    return next;
  }

  function clearStored(storage, fileName) {
    const key = fileKey(fileName);
    if (!key || !storage) return;
    const all = readStore(storage);
    delete all[key];
    writeStore(storage, all);
  }

  const api = {
    readLimits: readLimits,
    readExpects: readExpects,
    applyLimit: applyLimit,
    applyExpect: applyExpect,
    remember: remember,
    restore: restore,
    clearStored: clearStored,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevTol = api;
})();
