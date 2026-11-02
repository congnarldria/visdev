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

  function statusText(running, editor, line, active, paused) {
    if (running == null || sameProgram(running, editor)) return "";
    const where = line ? `第 ${line} 行 · ` : "";
    if (active) return `${where}正在跑按下執行時的程式，下一輪執行或單步才用現在的內容`;
    if (paused === "breakpoint") return `${where}中斷點在按下執行時的程式，再按執行或單步會改跑現在的內容`;
    if (paused === "stopped" || paused === "stop") return "停掉的是按下執行時的程式，再按執行或單步才用現在的內容";
    return "跑完的是按下執行時的程式，再按執行或單步才用現在的內容";
  }

  const api = { sameProgram: sameProgram, arrowLine: arrowLine, statusText: statusText };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevRun = api;
})();
