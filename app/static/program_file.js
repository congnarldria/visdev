(function () {
  function savePlan(target, saveAs) {
    const name = (target && target.name) || "program.viz";
    if (saveAs) return { mode: "ask", name: name };
    if (target && target.kind === "file" && target.path) return { mode: "file", path: target.path, name: name };
    if (target && target.kind === "workspace" && target.name) return { mode: "workspace", name: target.name };
    return {
      mode: "workspace",
      name: name,
      copy: !!(target && (target.kind === "sample" || target.kind === "import")),
    };
  }

  const api = { savePlan: savePlan };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevProgram = api;
})();
