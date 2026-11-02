(function () {
  function createEditorHistory(options) {
    const gap = options && options.gap != null ? options.gap : 800;
    const limit = options && options.limit != null ? options.limit : 200;
    const now = (options && options.now) || function () { return Date.now(); };
    let past = [];
    let future = [];
    let current = { value: "", start: 0, end: 0 };
    let typing = false;
    let stamp = 0;
    let composing = false;
    let composeBase = null;

    function snap(value, start, end) {
      return { value: String(value), start: start || 0, end: end == null ? (start || 0) : end };
    }

    function burst(prev, next) {
      if (!prev || !next || prev.start !== prev.end || next.start !== next.end) return false;
      const grew = next.value.length === prev.value.length + 1;
      const shrank = next.value.length === prev.value.length - 1;
      if (grew) {
        return next.value.slice(0, prev.start) === prev.value.slice(0, prev.start)
          && next.value.slice(next.start) === prev.value.slice(prev.start)
          && next.start === prev.start + 1;
      }
      if (shrank) {
        return next.start === prev.start - 1
          && next.value === prev.value.slice(0, next.start) + prev.value.slice(prev.start);
      }
      return false;
    }

    function commit(value, start, end, mode) {
      if (composing) {
        current = snap(value, start, end);
        return current;
      }
      const next = snap(value, start, end);
      if (next.value === current.value) {
        current.start = next.start;
        current.end = next.end;
        return current;
      }
      const fresh = now();
      if (mode !== "checkpoint" && typing && fresh - stamp <= gap && burst(current, next)) {
        current = next;
        future = [];
        stamp = fresh;
        return current;
      }
      past.push(current);
      if (past.length > limit) past.shift();
      current = next;
      future = [];
      typing = mode !== "checkpoint" && burst(past[past.length - 1], next);
      stamp = fresh;
      return current;
    }

    function undo() {
      if (!past.length) return current;
      future.push(current);
      current = past.pop();
      typing = false;
      return current;
    }

    function redo() {
      if (!future.length) return current;
      past.push(current);
      current = future.pop();
      typing = false;
      return current;
    }

    function reset(value, start, end) {
      past = [];
      future = [];
      current = snap(value, start, end);
      typing = false;
      composing = false;
      composeBase = null;
      stamp = now();
      return current;
    }

    function beginComposition() {
      if (composing) return;
      composing = true;
      composeBase = current;
    }

    function endComposition(value, start, end) {
      const base = composeBase || current;
      composing = false;
      composeBase = null;
      current = base;
      typing = false;
      return commit(value, start, end, "checkpoint");
    }

    function canUndo() { return past.length > 0; }
    function canRedo() { return future.length > 0; }

    return {
      commit: commit,
      undo: undo,
      redo: redo,
      reset: reset,
      beginComposition: beginComposition,
      endComposition: endComposition,
      canUndo: canUndo,
      canRedo: canRedo,
    };
  }

  const api = { createEditorHistory: createEditorHistory };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevEdit = api;
})();
