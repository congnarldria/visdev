(function () {
  function sameProgram(running, editor) {
    return String(running == null ? "" : running) === String(editor == null ? "" : editor);
  }

  function arrowLine(running, editor, line) {
    const number = Number(line);
    if (!number) return null;
    if (running == null) return number;
    const runningLines = String(running).split("\n");
    const editorLines = String(editor == null ? "" : editor).split("\n");
    if (runningLines[number - 1] == null) return null;
    if (editorLines[number - 1] === runningLines[number - 1]) return number;
    return null;
  }

  const RESULT_FIELDS = [
    ["Judge", "判定"],
    ["Diameter", "直徑"],
    ["Width", "寬"],
    ["Height", "高"],
    ["Gap", "間距"],
    ["BaselineX", "基線"],
    ["Text", "條碼"],
    ["QrText", "二維碼"],
    ["Count", "個數"],
    ["BrightCount", "亮點"],
    ["Score", "最低分"],
    ["Area", "面積"],
    ["BrightArea", "亮點面積"],
  ];

  function compactValue(text, name) {
    let raw = String(text == null ? "" : text).trim();
    if (!raw) return "";
    if (raw.startsWith("[")) {
      const body = raw.replace(/^\[/, "").replace(/\](?:\s*\(\d+\))?$/, "");
      const parts = body.split(",").map((item) => item.trim()).filter((item) => item && item !== "…");
      if (name === "Score" && parts.length) {
        const values = parts.map(Number).filter((item) => Number.isFinite(item));
        if (!values.length) return "";
        return compactValue(String(Math.min.apply(null, values)), "");
      }
      if (parts.length !== 1) return "";
      raw = parts[0];
    }
    if ((raw.startsWith("'") && raw.endsWith("'")) || (raw.startsWith('"') && raw.endsWith('"'))) {
      raw = raw.slice(1, -1);
    }
    if (/^-?\d+\.\d+$/.test(raw)) {
      const number = Number(raw);
      if (Number.isFinite(number)) {
        const rounded = Math.round(number * 100) / 100;
        raw = String(rounded);
      }
    }
    return raw;
  }

  function resultSummary(control) {
    const byName = new Map((control || []).map((item) => [item.name, item]));
    const parts = [];
    RESULT_FIELDS.forEach((field) => {
      const item = byName.get(field[0]);
      if (!item) return;
      const shown = compactValue(item.text, field[0]);
      if (!shown) return;
      parts.push(field[1] + " " + shown);
    });
    return parts.slice(0, 5).join(" · ");
  }

  function statusText(running, editor, line, active, paused) {
    if (running == null || sameProgram(running, editor)) return "";
    const where = line ? `第 ${line} 行 · ` : "";
    if (active) return `${where}正在跑按下執行時的程式，下一輪執行或單步才用現在的內容`;
    if (paused === "breakpoint") return `${where}中斷點在按下執行時的程式，再按執行或單步會改跑現在的內容`;
    if (paused === "stopped" || paused === "stop") return "停掉的是按下執行時的程式，再按執行或單步才用現在的內容";
    return "跑完的是按下執行時的程式，再按執行或單步才用現在的內容";
  }

  const api = { sameProgram: sameProgram, arrowLine: arrowLine, statusText: statusText, resultSummary: resultSummary };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevRun = api;
})();
