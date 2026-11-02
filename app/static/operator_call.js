(function () {
  function splitCall(text) {
    const parts = [];
    let buf = "";
    let quote = "";
    let depth = 0;
    for (const ch of String(text || "")) {
      if (quote) {
        buf += ch;
        if (ch === quote) quote = "";
        continue;
      }
      if (ch === "'" || ch === "\"") {
        quote = ch;
        buf += ch;
        continue;
      }
      if (ch === "[" || ch === "(") {
        depth += 1;
        buf += ch;
        continue;
      }
      if (ch === "]" || ch === ")") {
        depth -= 1;
        buf += ch;
        continue;
      }
      if (ch === "," && depth === 0) {
        parts.push(buf.trim());
        buf = "";
        continue;
      }
      buf += ch;
    }
    if (buf.trim()) parts.push(buf.trim());
    return parts;
  }

  function exampleTokens(op) {
    const example = op && op.example ? String(op.example) : "";
    const open = example.indexOf("(");
    const close = example.lastIndexOf(")");
    if (open < 0 || close < open) return [];
    return splitCall(example.slice(open + 1, close));
  }

  function fallbackValue(param) {
    if (param.role === "控制輸入") return param.default || "";
    return param.name || "";
  }

  function callValues(op, tokens) {
    const source = tokens === undefined ? exampleTokens(op) : tokens;
    return (op.params || []).map((param, index) => {
      const value = source[index];
      return value ? value : fallbackValue(param);
    });
  }

  function formatCall(name, values) {
    return `${name} (${(values || []).join(", ")})`;
  }

  function parseLine(text) {
    const src = String(text || "").trim();
    if (!src || src.startsWith("*")) return null;
    const match = src.match(/^((?:Vizion\.)?[A-Za-z_][A-Za-z0-9_]*)\s*\(/i);
    if (!match) return null;
    let quote = "";
    let depth = 1;
    let body = "";
    for (let i = match[0].length; i < src.length; i += 1) {
      const ch = src[i];
      if (quote) {
        body += ch;
        if (ch === quote) quote = "";
        continue;
      }
      if (ch === "'" || ch === '"') {
        quote = ch;
        body += ch;
        continue;
      }
      if (ch === "(") {
        depth += 1;
        body += ch;
        continue;
      }
      if (ch === ")") {
        depth -= 1;
        if (depth === 0) return { name: match[1], args: splitCall(body) };
        body += ch;
        continue;
      }
      body += ch;
    }
    return null;
  }

  function memoryRecord(op, values) {
    const fields = {};
    (op.params || []).forEach((param, index) => {
      const value = String(values[index] == null ? "" : values[index]).trim();
      if (value) fields[param.name] = value;
    });
    return fields;
  }

  function valuesWithMemory(op, saved) {
    const values = callValues(op);
    if (!saved) return values;
    (op.params || []).forEach((param, index) => {
      if (saved[param.name]) values[index] = saved[param.name];
    });
    return values;
  }

  function dialogValues(op, options) {
    const lineIndex = options && options.lineIndex;
    const editing = Number.isInteger(lineIndex);
    const preset = options && options.preset;
    let values;
    if (preset) values = preset.slice();
    else if (editing) values = callValues(op);
    else values = valuesWithMemory(op, options && options.memory);
    return {
      editing: editing,
      label: editing ? "更新" : "插入",
      values: values,
      call: formatCall(op.name, values),
    };
  }

  function replaceLine(source, lineIndex, call) {
    const lines = String(source).split("\n");
    const current = lines[lineIndex] == null ? "" : lines[lineIndex];
    const indent = (current.match(/^\s*/) || [""])[0];
    lines[lineIndex] = indent + call;
    return {
      text: lines.join("\n"),
      caret: lines.slice(0, lineIndex + 1).join("\n").length,
      call: call,
    };
  }

  const api = {
    splitCall: splitCall,
    exampleTokens: exampleTokens,
    callValues: callValues,
    formatCall: formatCall,
    parseLine: parseLine,
    memoryRecord: memoryRecord,
    valuesWithMemory: valuesWithMemory,
    dialogValues: dialogValues,
    replaceLine: replaceLine,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevCall = api;
})();
