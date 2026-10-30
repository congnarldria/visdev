/* Dockable panes: drag a grip onto another pane to dock, or drop outside to float. */
const VisDevDock = (() => {
  const KEY = "visdev.dock";
  const TITLES = { program: "程式", gfx: "圖形視窗", vars: "變數", bottom: "算子" };
  const ORDER = ["program", "gfx", "vars", "bottom"];

  const dock = document.getElementById("dock");
  const panes = {};
  document.querySelectorAll("[data-dock]").forEach((node) => {
    panes[node.dataset.dock] = node;
  });

  let tree = null;
  let floats = [];
  let autoHide = {};
  let flyout = null;
  let flySticky = false;
  let nodesByKey = new Map();
  let keySeq = 1;
  let suppressClick = false;
  let preview = null;

  function defaultTree() {
    return {
      type: "split",
      dir: "column",
      ratio: 0.72,
      first: {
        type: "split",
        dir: "row",
        ratio: 0.32,
        first: tabs("program"),
        second: {
          type: "split",
          dir: "row",
          ratio: 0.68,
          first: tabs("gfx"),
          second: tabs("vars"),
        },
      },
      second: tabs("bottom"),
    };
  }

  function tabs(id) {
    return { type: "tabs", active: id, panes: [id] };
  }

  function walk(node, fn, parent = null, key = null) {
    if (!node) return;
    fn(node, parent, key);
    if (node.type === "split") {
      walk(node.first, fn, node, "first");
      walk(node.second, fn, node, "second");
    }
  }

  function roots() {
    const list = [];
    if (tree) list.push({ kind: "main", node: tree });
    floats.forEach((item) => {
      if (item.node) list.push({ kind: "float", float: item, node: item.node });
    });
    return list;
  }

  function findRoot(target) {
    for (const root of roots()) {
      let hit = false;
      walk(root.node, (node) => {
        if (node === target) hit = true;
      });
      if (hit) return root;
    }
    return null;
  }

  function findTabs(id) {
    let found = null;
    roots().forEach((root) => {
      walk(root.node, (node) => {
        if (node.type === "tabs" && node.panes.includes(id)) found = node;
      });
    });
    return found;
  }

  function topNode(root) {
    return root.kind === "main" ? tree : root.float.node;
  }

  function setTop(root, value) {
    if (root.kind === "main") tree = value;
    else root.float.node = value;
    if (root.kind === "float" && !value) floats = floats.filter((item) => item !== root.float);
  }

  function parentOf(target, top) {
    let found = null;
    walk(top, (node, parent, key) => {
      if (node === target) found = { parent, key };
    });
    return found;
  }

  function replaceNode(target, next) {
    const root = findRoot(target);
    if (!root) return;
    const top = topNode(root);
    if (top === target) {
      setTop(root, next);
      return;
    }
    const loc = parentOf(target, top);
    if (loc && loc.parent) loc.parent[loc.key] = next;
  }

  function removeNode(target) {
    const root = findRoot(target);
    if (!root) return;
    const top = topNode(root);
    if (top === target) {
      setTop(root, null);
      return;
    }
    const loc = parentOf(target, top);
    if (!loc || !loc.parent) return;
    const other = loc.key === "first" ? loc.parent.second : loc.parent.first;
    replaceNode(loc.parent, other);
  }

  function detach(id) {
    const group = findTabs(id);
    if (!group) return;
    group.panes = group.panes.filter((item) => item !== id);
    if (group.active === id) group.active = group.panes[0] || null;
    if (!group.panes.length) removeNode(group);
  }

  function insertAt(target, id, zone) {
    if (!target || !findRoot(target)) {
      tree = tabs(id);
      return;
    }
    if (zone === "center") {
      if (!target.panes.includes(id)) target.panes.push(id);
      target.active = id;
      return;
    }
    const dir = zone === "left" || zone === "right" ? "row" : "column";
    const created = tabs(id);
    const split = {
      type: "split",
      dir,
      ratio: 0.5,
      first: zone === "left" || zone === "top" ? created : target,
      second: zone === "left" || zone === "top" ? target : created,
    };
    replaceNode(target, split);
  }

  function collect(node, into) {
    walk(node, (item) => {
      if (item.type === "tabs") into.push(...item.panes);
    });
  }

  function layoutOk(state) {
    if (!state || typeof state !== "object") return false;
    const ids = [];
    collect(state.tree, ids);
    (state.floats || []).forEach((item) => collect(item.node, ids));
    const hidden = Object.keys(state.autoHide || {}).filter((id) => ORDER.includes(id) && ["left", "right", "top", "bottom"].includes(state.autoHide[id]));
    const all = ids.concat(hidden);
    return ORDER.every((id) => all.filter((item) => item === id).length === 1) && all.length === ORDER.length;
  }

  function load() {
    try {
      const state = JSON.parse(localStorage.getItem(KEY) || "null");
      if (layoutOk(state)) {
        tree = state.tree;
        floats = state.floats || [];
        autoHide = state.autoHide || {};
        return;
      }
    } catch (error) {
      /* keep the default layout */
    }
    tree = defaultTree();
    floats = [];
    autoHide = {};
  }

  function save() {
    localStorage.setItem(KEY, JSON.stringify({ tree, floats, autoHide }));
  }

  function ensureChrome(pane) {
    const bar = pane.querySelector(".pane-bar, .tabs");
    if (!bar || bar.querySelector(".dock-grip")) return;
    const grip = document.createElement("button");
    grip.type = "button";
    grip.className = "dock-grip";
    grip.title = "拖曳停靠";
    grip.textContent = "⋮⋮";
    grip.addEventListener("mousedown", (event) => {
      event.preventDefault();
      event.stopPropagation();
      beginDrag(pane.dataset.dock, event);
    });
    bar.prepend(grip);
    const tools = document.createElement("span");
    tools.className = "dock-tools";
    const pin = document.createElement("button");
    pin.type = "button";
    pin.className = "dock-pin on";
    pin.title = "自動隱藏";
    pin.innerHTML = '<svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true"><path fill="currentColor" d="M5.2 1.2h5.6v1.4H9.2v2.1l2.3 2.2v1.3H8.7v6.6H7.3V8.2H4.5V6.9l2.3-2.2V2.6H5.2V1.2z"/></svg>';
    pin.addEventListener("mousedown", (event) => event.stopPropagation());
    pin.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const id = pane.dataset.dock;
      if (autoHide[id]) pinBack(id);
      else unpin(id);
    });
    const back = document.createElement("button");
    back.type = "button";
    back.className = "dock-back";
    back.textContent = "收回";
    back.hidden = true;
    back.addEventListener("click", () => dockBack(pane.dataset.dock));
    tools.append(pin, back);
    bar.append(tools);
  }

  function renderNode(node) {
    if (node.type === "tabs") {
      const group = document.createElement("div");
      const key = `g${keySeq++}`;
      nodesByKey.set(key, node);
      group.className = "dock-group";
      group.dataset.key = key;
      group.dataset.panes = node.panes.join(",");
      if (!node.panes.includes(node.active)) node.active = node.panes[0];
      if (node.panes.length > 1) {
        const strip = document.createElement("div");
        strip.className = "dock-tabstrip";
        node.panes.forEach((id) => {
          const tab = document.createElement("button");
          tab.type = "button";
          tab.className = `dock-tab${id === node.active ? " on" : ""}`;
          tab.textContent = TITLES[id] || id;
          tab.addEventListener("mousedown", (event) => {
            if (event.button === 0) beginDrag(id, event);
          });
          tab.addEventListener("click", () => {
            if (suppressClick) return;
            node.active = id;
            save();
            render();
          });
          strip.append(tab);
        });
        group.append(strip);
      }
      node.panes.forEach((id) => {
        const pane = panes[id];
        pane.hidden = id !== node.active;
        group.append(pane);
      });
      return group;
    }
    const split = document.createElement("div");
    split.className = `dock-split ${node.dir === "row" ? "row" : "column"}`;
    const first = document.createElement("div");
    first.className = "dock-child";
    first.style.flex = `${node.ratio} 1 0`;
    first.append(renderNode(node.first));
    const bar = document.createElement("div");
    bar.className = "dock-bar";
    bar.addEventListener("mousedown", (event) => beginResize(node, split, event));
    const second = document.createElement("div");
    second.className = "dock-child";
    second.style.flex = `${1 - node.ratio} 1 0`;
    second.append(renderNode(node.second));
    split.append(first, bar, second);
    return split;
  }

  function renderFloat(item) {
    const wrap = document.createElement("section");
    wrap.className = "dock-float";
    wrap.style.left = `${Math.max(0, item.x)}px`;
    wrap.style.top = `${Math.max(0, item.y)}px`;
    wrap.style.width = `${item.w}px`;
    wrap.style.height = `${item.h}px`;
    const bar = document.createElement("div");
    bar.className = "dock-floatbar";
    const label = document.createElement("span");
    label.textContent = floatTitle(item.node);
    const back = document.createElement("button");
    back.type = "button";
    back.textContent = "收回";
    back.addEventListener("click", () => dockNode(item));
    bar.append(label, back);
    bar.addEventListener("mousedown", (event) => {
      if (event.target.closest("button")) return;
      moveFloat(item, wrap, event);
    });
    const body = document.createElement("div");
    body.className = "dock-floatbody";
    if (item.node) body.append(renderNode(item.node));
    const handle = document.createElement("div");
    handle.className = "dock-resize";
    handle.addEventListener("mousedown", (event) => resizeFloat(item, wrap, event));
    wrap.append(bar, body, handle);
    wrap.addEventListener("mousedown", () => {
      wrap.style.zIndex = String(++keySeq);
    });
    return wrap;
  }

  function floatTitle(node) {
    const ids = [];
    collect(node, ids);
    return ids.map((id) => TITLES[id] || id).join(" · ") || "浮動視窗";
  }

  function render() {
    nodesByKey = new Map();
    ORDER.forEach((id) => ensureChrome(panes[id]));
    const root = document.createElement("div");
    root.className = "dock-root";
    if (tree) root.append(renderNode(tree));
    const hosted = [root, ...floats.filter((item) => item.node).map(renderFloat)];
    dock.replaceChildren(...hosted);
    const edges = new Set(Object.values(autoHide));
    root.style.paddingLeft = edges.has("left") ? "26px" : "";
    root.style.paddingRight = edges.has("right") ? "26px" : "";
    root.style.paddingTop = edges.has("top") ? "26px" : "";
    root.style.paddingBottom = edges.has("bottom") ? "26px" : "";
    ORDER.forEach((id) => {
      const host = panes[id].closest(".dock-float");
      const back = panes[id].querySelector(".dock-back");
      if (back) back.hidden = !host || host.querySelectorAll("[data-dock]").length < 2;
      const pin = panes[id].querySelector(".dock-pin");
      if (pin) {
        const hidden = Boolean(autoHide[id]);
        pin.classList.toggle("on", !hidden);
        pin.title = hidden ? "釘選" : "自動隱藏";
      }
    });
    document.querySelectorAll("[data-pane]").forEach((item) => {
      item.setAttribute("aria-checked", autoHide[item.dataset.pane] ? "false" : "true");
    });
    renderRails();
    window.dispatchEvent(new Event("visdev-layout"));
  }

  function edgeOf(id) {
    const rect = panes[id].getBoundingClientRect();
    const host = dock.getBoundingClientRect();
    if (rect.width < 2 || rect.height < 2) return "right";
    const cx = rect.left + rect.width / 2 - host.left;
    const cy = rect.top + rect.height / 2 - host.top;
    const dx = Math.min(cx, host.width - cx);
    const dy = Math.min(cy, host.height - cy);
    if (dx <= dy) return cx < host.width / 2 ? "left" : "right";
    return cy < host.height / 2 ? "top" : "bottom";
  }

  function attachEdge(id, edge) {
    const created = tabs(id);
    if (!tree) {
      tree = created;
      return;
    }
    const horizontal = edge === "left" || edge === "right";
    const before = edge === "left" || edge === "top";
    tree = {
      type: "split",
      dir: horizontal ? "row" : "column",
      ratio: before ? 0.3 : 0.7,
      first: before ? created : tree,
      second: before ? tree : created,
    };
  }

  function unpin(id) {
    if (autoHide[id]) return;
    const edge = edgeOf(id);
    detach(id);
    autoHide[id] = edge;
    flyout = id;
    flySticky = true;
    save();
    render();
  }

  function pinBack(id, skipRender) {
    if (!autoHide[id]) return;
    const edge = autoHide[id];
    delete autoHide[id];
    if (flyout === id) {
      flyout = null;
      flySticky = false;
    }
    attachEdge(id, edge);
    if (skipRender) return;
    save();
    render();
  }

  function pinAll() {
    ORDER.forEach((id) => pinBack(id, true));
    save();
    render();
  }

  function renderRails() {
    dock.querySelectorAll(".dock-rail, .dock-fly, .dock-park").forEach((node) => node.remove());
    const park = document.createElement("div");
    park.className = "dock-park";
    park.hidden = true;
    const groups = { left: [], right: [], top: [], bottom: [] };
    ORDER.forEach((id) => {
      const edge = autoHide[id];
      if (!edge) return;
      groups[edge].push(id);
      if (id !== flyout) {
        panes[id].hidden = true;
        park.append(panes[id]);
      }
    });
    dock.append(park);
    ["left", "right", "top", "bottom"].forEach((edge) => {
      if (!groups[edge].length) return;
      const rail = document.createElement("div");
      rail.className = `dock-rail ${edge}`;
      groups[edge].forEach((id) => {
        const tab = document.createElement("button");
        tab.type = "button";
        tab.className = `rail-tab${flyout === id ? " on" : ""}`;
        tab.dataset.pane = id;
        tab.textContent = TITLES[id] || id;
        tab.title = "顯示。圖釘可以把它釘回版面";
        tab.addEventListener("mouseenter", () => {
          window.clearTimeout(scheduleFlyClose.timer);
          window.clearTimeout(openFly.timer);
          openFly.timer = window.setTimeout(() => openFly(id, false), 250);
        });
        tab.addEventListener("mouseleave", () => {
          window.clearTimeout(openFly.timer);
          scheduleFlyClose();
        });
        tab.addEventListener("click", () => {
          window.clearTimeout(openFly.timer);
          if (flyout === id && flySticky) closeFly();
          else openFly(id, true);
        });
        rail.append(tab);
      });
      dock.append(rail);
    });
    showFly();
  }

  function openFly(id, sticky) {
    if (!autoHide[id]) return;
    if (sticky) flySticky = true;
    if (!sticky && flySticky && flyout && flyout !== id) return;
    flyout = id;
    showFly();
  }

  function closeFly() {
    if (!flyout) return;
    const id = flyout;
    flyout = null;
    flySticky = false;
    const park = dock.querySelector(".dock-park");
    if (park) {
      panes[id].hidden = true;
      park.append(panes[id]);
    }
    const fly = dock.querySelector(".dock-fly");
    if (fly) fly.remove();
    dock.querySelectorAll(".rail-tab.on").forEach((tab) => tab.classList.remove("on"));
  }

  function scheduleFlyClose() {
    window.clearTimeout(scheduleFlyClose.timer);
    scheduleFlyClose.timer = window.setTimeout(() => {
      if (!flySticky) closeFly();
    }, 400);
  }

  function showFly() {
    const previous = dock.querySelector(".dock-fly");
    if (!flyout || !autoHide[flyout]) {
      if (previous) previous.remove();
      return;
    }
    const fly = previous || document.createElement("div");
    fly.className = `dock-fly ${autoHide[flyout]}`;
    panes[flyout].hidden = false;
    fly.append(panes[flyout]);
    if (!previous) {
      fly.addEventListener("mouseenter", () => window.clearTimeout(scheduleFlyClose.timer));
      fly.addEventListener("mouseleave", scheduleFlyClose);
      dock.append(fly);
    }
    dock.querySelectorAll(".rail-tab").forEach((tab) => {
      tab.classList.toggle("on", tab.dataset.pane === flyout);
    });
    window.dispatchEvent(new Event("visdev-layout"));
  }

  function beginResize(node, split, event) {
    event.preventDefault();
    const rect = split.getBoundingClientRect();
    const move = (ev) => {
      const pos = node.dir === "row"
        ? (ev.clientX - rect.left) / rect.width
        : (ev.clientY - rect.top) / rect.height;
      node.ratio = Math.min(0.82, Math.max(0.18, pos));
      split.children[0].style.flex = `${node.ratio} 1 0`;
      split.children[2].style.flex = `${1 - node.ratio} 1 0`;
      window.dispatchEvent(new Event("visdev-layout"));
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      save();
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function zoneAt(rect, x, y) {
    const rx = (x - rect.left) / Math.max(rect.width, 1);
    const ry = (y - rect.top) / Math.max(rect.height, 1);
    if (rx > 0.3 && rx < 0.7 && ry > 0.3 && ry < 0.7) return "center";
    const distance = { left: rx, right: 1 - rx, top: ry, bottom: 1 - ry };
    return Object.entries(distance).sort((a, b) => a[1] - b[1])[0][0];
  }

  function showPreview(group, zone) {
    if (!preview) {
      preview = document.createElement("div");
      preview.className = "dock-preview";
      document.body.append(preview);
    }
    const rect = group.getBoundingClientRect();
    const box = { left: rect.left, top: rect.top, width: rect.width, height: rect.height };
    if (zone === "left" || zone === "right") box.width = rect.width / 2;
    if (zone === "top" || zone === "bottom") box.height = rect.height / 2;
    if (zone === "right") box.left = rect.left + rect.width / 2;
    if (zone === "bottom") box.top = rect.top + rect.height / 2;
    if (zone === "center") {
      box.left = rect.left + rect.width * 0.18;
      box.top = rect.top + rect.height * 0.18;
      box.width = rect.width * 0.64;
      box.height = rect.height * 0.64;
    }
    preview.hidden = false;
    preview.style.left = `${box.left}px`;
    preview.style.top = `${box.top}px`;
    preview.style.width = `${box.width}px`;
    preview.style.height = `${box.height}px`;
  }

  function hidePreview() {
    if (preview) preview.hidden = true;
  }

  function groupAt(x, y) {
    const node = document.elementFromPoint(x, y);
    return node && node.closest("[data-key]");
  }

  function beginDrag(id, event) {
    const startX = event.clientX;
    const startY = event.clientY;
    let moved = false;
    const move = (ev) => {
      if (Math.hypot(ev.clientX - startX, ev.clientY - startY) < 5) return;
      moved = true;
      const group = groupAt(ev.clientX, ev.clientY);
      if (!group) {
        hidePreview();
        return;
      }
      showPreview(group, zoneAt(group.getBoundingClientRect(), ev.clientX, ev.clientY));
    };
    const up = (ev) => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      hidePreview();
      if (!moved) return;
      suppressClick = true;
      setTimeout(() => {
        suppressClick = false;
      }, 0);
      const group = groupAt(ev.clientX, ev.clientY);
      if (!group) {
        const host = dock.getBoundingClientRect();
        floatPane(id, ev.clientX - host.left - 120, ev.clientY - host.top - 18);
        return;
      }
      const target = nodesByKey.get(group.dataset.key);
      const zone = zoneAt(group.getBoundingClientRect(), ev.clientX, ev.clientY);
      if (!target) return;
      if (target.panes.length === 1 && target.panes[0] === id) return;
      if (target.panes.includes(id) && zone === "center") return;
      detach(id);
      insertAt(target, id, zone);
      save();
      render();
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function floatPane(id, x, y) {
    detach(id);
    const host = dock.getBoundingClientRect();
    floats.push({
      node: tabs(id),
      x: Math.max(8, Math.min(x, host.width - 280)),
      y: Math.max(8, Math.min(y, host.height - 160)),
      w: Math.min(640, Math.max(320, host.width * 0.42)),
      h: Math.min(480, Math.max(220, host.height * 0.55)),
    });
    save();
    render();
  }

  function dockBack(id) {
    const group = findTabs(id);
    if (!group) return;
    detach(id);
    const anchor = firstTabs(tree);
    if (!anchor) tree = tabs(id);
    else insertAt(anchor, id, "right");
    save();
    render();
  }

  function dockNode(item) {
    const ids = [];
    collect(item.node, ids);
    floats = floats.filter((entry) => entry !== item);
    ids.forEach((id) => {
      const anchor = firstTabs(tree);
      if (!anchor) tree = tabs(id);
      else insertAt(anchor, id, "right");
    });
    save();
    render();
  }

  function firstTabs(node) {
    let found = null;
    walk(node, (item) => {
      if (!found && item.type === "tabs") found = item;
    });
    return found;
  }

  function moveFloat(item, wrap, event) {
    event.preventDefault();
    const originX = event.clientX - item.x;
    const originY = event.clientY - item.y;
    const move = (ev) => {
      item.x = Math.max(0, ev.clientX - originX);
      item.y = Math.max(0, ev.clientY - originY);
      wrap.style.left = `${item.x}px`;
      wrap.style.top = `${item.y}px`;
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      save();
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function resizeFloat(item, wrap, event) {
    event.preventDefault();
    event.stopPropagation();
    const originX = event.clientX;
    const originY = event.clientY;
    const width = item.w;
    const height = item.h;
    const move = (ev) => {
      item.w = Math.max(280, width + ev.clientX - originX);
      item.h = Math.max(180, height + ev.clientY - originY);
      wrap.style.width = `${item.w}px`;
      wrap.style.height = `${item.h}px`;
      window.dispatchEvent(new Event("visdev-layout"));
    };
    const up = () => {
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
      save();
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  }

  function reset() {
    localStorage.removeItem(KEY);
    tree = defaultTree();
    floats = [];
    autoHide = {};
    flyout = null;
    flySticky = false;
    save();
    render();
  }

  load();
  ORDER.forEach((id) => ensureChrome(panes[id]));
  render();
  const resetButton = document.getElementById("btnLayout");
  if (resetButton) resetButton.onclick = reset;

  let menuOpen = null;
  let openedByHover = false;
  function closeMenus() {
    document.querySelectorAll(".menu-item.open").forEach((item) => {
      item.classList.remove("open");
      const drop = item.querySelector(".menu-drop");
      if (drop) drop.hidden = true;
    });
    document.querySelectorAll(".menu-fly").forEach((fly) => { fly.hidden = true; });
    menuOpen = null;
    openedByHover = false;
  }
  function openMenu(item) {
    closeMenus();
    item.classList.add("open");
    const drop = item.querySelector(".menu-drop");
    if (drop) drop.hidden = false;
    menuOpen = item;
  }
  document.querySelectorAll(".menubar > .menu-item > button").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      const item = button.parentElement;
      if (item.classList.contains("open")) {
        if (openedByHover) {
          openedByHover = false;
          return;
        }
        closeMenus();
      } else {
        openMenu(item);
        openedByHover = false;
      }
    });
    button.addEventListener("mouseenter", () => {
      if (!menuOpen) return;
      openMenu(button.parentElement);
      openedByHover = true;
    });
  });
  const menuBar = document.querySelector(".menubar");
  if (menuBar) {
    menuBar.addEventListener("mouseover", (event) => {
      const sub = event.target.closest(".menu-sub");
      if (!sub || !sub.parentElement) return;
      sub.parentElement.querySelectorAll(".menu-sub").forEach((node) => {
        const fly = node.querySelector(".menu-fly");
        if (fly) fly.hidden = node !== sub;
      });
    });
    menuBar.addEventListener("click", (event) => {
      const item = event.target.closest(".menu-drop button");
      if (!item) return;
      if (item.dataset.submenu) {
        const sub = item.parentElement;
        sub.parentElement.querySelectorAll(".menu-fly").forEach((node) => { node.hidden = true; });
        const fly = sub.querySelector(".menu-fly");
        if (fly) fly.hidden = false;
        return;
      }
      if (item.dataset.sample && window.VisDevCommands && window.VisDevCommands.openSample) {
        window.VisDevCommands.openSample(item.dataset.sample);
        closeMenus();
        return;
      }
      if (item.dataset.op && window.VisDevCommands && window.VisDevCommands.openOperator) {
        window.VisDevCommands.openOperator(item.dataset.op);
        closeMenus();
        return;
      }
      if (item.dataset.cmd && window.VisDevCommands && window.VisDevCommands[item.dataset.cmd]) {
        window.VisDevCommands[item.dataset.cmd](item);
      } else if (item.dataset.click) document.getElementById(item.dataset.click)?.click();
      if (item.dataset.pane) {
        const id = item.dataset.pane;
        if (autoHide[id]) pinBack(id);
        else unpin(id);
      }
      if (item.id === "menuPinAll") pinAll();
      closeMenus();
    });
  }
  document.addEventListener("mousedown", (event) => {
    if (!event.target.closest(".menubar")) closeMenus();
    if (!flyout || !flySticky) return;
    if (event.target.closest(".dock-fly, .rail-tab, .dock-pin")) return;
    closeFly();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    closeMenus();
    closeFly();
  });

  return { floatPane, dockBack, reset, unpin, pinBack, pinAll, isAutoHide: (id) => Boolean(autoHide[id]), movePane: (id, targetId, zone) => {
    const target = findTabs(targetId);
    if (!target) return;
    detach(id);
    insertAt(target, id, zone);
    save();
    render();
  } };
})();
