(function () {
  function pointInPoly(x, y, points) {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
      const xi = points[i][0];
      const yi = points[i][1];
      const xj = points[j][0];
      const yj = points[j][1];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function lineDistance(x, y, points) {
    let best = Infinity;
    for (let i = 1; i < points.length; i += 1) {
      const ax = points[i - 1][0];
      const ay = points[i - 1][1];
      const bx = points[i][0];
      const by = points[i][1];
      const dx = bx - ax;
      const dy = by - ay;
      const len2 = dx * dx + dy * dy || 1;
      let t = ((x - ax) * dx + (y - ay) * dy) / len2;
      t = Math.max(0, Math.min(1, t));
      best = Math.min(best, Math.hypot(x - (ax + t * dx), y - (ay + t * dy)));
    }
    return best;
  }

  function loadImg(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("image"));
      image.src = url;
    });
  }

  function mount(canvas, hooks) {
    const ctx = canvas.getContext("2d");
    const cam = { x: 0, y: 0, scale: 1, user: false, fitW: 0, fitH: 0 };
    let mode = "select";
    let disp = null;
    let baseImage = null;
    let selection = null;
    let shownKey = "";
    let drag = null;
    let zoomBox = null;
    let dpr = 1;
    const images = new Map();

    function imagePoint(event) {
      const rect = canvas.getBoundingClientRect();
      return {
        sx: event.clientX - rect.left,
        sy: event.clientY - rect.top,
        col: (event.clientX - rect.left - cam.x) / cam.scale,
        row: (event.clientY - rect.top - cam.y) / cam.scale,
      };
    }

    function resize() {
      const rect = canvas.getBoundingClientRect();
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
      draw();
    }

    function fit(markUser) {
      const rect = canvas.getBoundingClientRect();
      if (!disp || !disp.width || rect.width < 2) return;
      const pad = 12;
      cam.scale = Math.max(0.02, Math.min((rect.width - pad * 2) / disp.width, (rect.height - pad * 2) / disp.height));
      cam.x = (rect.width - disp.width * cam.scale) / 2;
      cam.y = (rect.height - disp.height * cam.scale) / 2;
      cam.user = !!markUser;
      draw();
    }

    function one() {
      const rect = canvas.getBoundingClientRect();
      if (!disp) return;
      cam.scale = 1;
      cam.x = (rect.width - disp.width) / 2;
      cam.y = (rect.height - disp.height) / 2;
      cam.user = true;
      draw();
    }

    function zoomBy(factor) {
      const rect = canvas.getBoundingClientRect();
      const mx = rect.width / 2;
      const my = rect.height / 2;
      const next = Math.min(48, Math.max(0.05, cam.scale * factor));
      const k = next / cam.scale;
      cam.x = mx - (mx - cam.x) * k;
      cam.y = my - (my - cam.y) * k;
      cam.scale = next;
      cam.user = true;
      draw();
    }

    function zoomToRect(col0, row0, col1, row1) {
      const left = Math.min(col0, col1);
      const top = Math.min(row0, row1);
      const width = Math.max(1, Math.abs(col1 - col0));
      const height = Math.max(1, Math.abs(row1 - row0));
      const rect = canvas.getBoundingClientRect();
      const pad = 8;
      cam.scale = Math.min(48, Math.max(0.05, Math.min((rect.width - pad * 2) / width, (rect.height - pad * 2) / height)));
      cam.x = (rect.width - width * cam.scale) / 2 - left * cam.scale;
      cam.y = (rect.height - height * cam.scale) / 2 - top * cam.scale;
      cam.user = true;
      draw();
    }

    function strokePicked() {
      if (!selection || !disp) return;
      const rings = (disp.shapes || []).filter((shape) => {
        if (shape.type !== "polyline" || !shape.points || shape.points.length < 2) return false;
        if (shape.source !== selection.name) return false;
        if ((shape.kind || "region") !== selection.kind) return false;
        if (selection.kind === "xld" && shape.group != null) return shape.group === selection.index;
        return shape.object === selection.index;
      });
      if (!rings.length) return;
      const drawRings = () => {
        rings.forEach((shape) => {
          ctx.beginPath();
          ctx.moveTo(shape.points[0][0], shape.points[0][1]);
          for (let i = 1; i < shape.points.length; i += 1) ctx.lineTo(shape.points[i][0], shape.points[i][1]);
          if (shape.closed !== false) ctx.closePath();
          ctx.stroke();
        });
      };
      ctx.save();
      ctx.setLineDash([7 / cam.scale, 5 / cam.scale]);
      ctx.lineJoin = "round";
      ctx.strokeStyle = "rgba(0,0,0,0.85)";
      ctx.lineWidth = 4 / cam.scale;
      drawRings();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2 / cam.scale;
      drawRings();
      ctx.restore();
    }

    function draw() {
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#101216";
      ctx.fillRect(0, 0, width, height);
      if (!disp) return;
      ctx.save();
      ctx.translate(cam.x, cam.y);
      ctx.scale(cam.scale, cam.scale);
      ctx.fillStyle = "#000";
      ctx.fillRect(0, 0, disp.width, disp.height);
      if (baseImage) {
        ctx.imageSmoothingEnabled = cam.scale < 3;
        ctx.drawImage(baseImage, 0, 0);
      }
      for (const shape of disp.shapes || []) {
        if (shape.type === "fill" && images.get(shape.image)) ctx.drawImage(images.get(shape.image), 0, 0);
      }
      for (const shape of disp.shapes || []) {
        if (shape.type === "polyline" && shape.points && shape.points.length && !shape.hit) {
          ctx.beginPath();
          ctx.moveTo(shape.points[0][0], shape.points[0][1]);
          for (let i = 1; i < shape.points.length; i += 1) ctx.lineTo(shape.points[i][0], shape.points[i][1]);
          if (shape.closed) ctx.closePath();
          ctx.strokeStyle = shape.color || "#00e05a";
          ctx.lineWidth = (shape.width || 2) / cam.scale;
          ctx.stroke();
        }
      }
      strokePicked();
      if (zoomBox) {
        ctx.strokeStyle = "#ffe14a";
        ctx.lineWidth = 1 / cam.scale;
        ctx.setLineDash([4 / cam.scale, 3 / cam.scale]);
        ctx.strokeRect(zoomBox.col0, zoomBox.row0, zoomBox.col1 - zoomBox.col0, zoomBox.row1 - zoomBox.row0);
      }
      ctx.restore();
    }

    function pickAt(col, row) {
      const shapes = disp && disp.shapes || [];
      let regionHit = null;
      let xldHit = null;
      let xldDist = Infinity;
      const limit = 12 / cam.scale;
      for (let i = shapes.length - 1; i >= 0; i -= 1) {
        const shape = shapes[i];
        if (shape.type !== "polyline" || shape.object == null || !shape.source) continue;
        if (shape.kind === "xld") {
          const dist = lineDistance(col, row, shape.points || []);
          if (dist <= limit && dist < xldDist) {
            xldDist = dist;
            xldHit = shape;
          }
        } else if (!regionHit && pointInPoly(col, row, shape.points || [])) {
          regionHit = shape;
        }
      }
      const shape = xldHit || regionHit;
      if (!shape) return null;
      const grouped = shape.kind === "xld" && shape.group != null;
      return {
        name: shape.source,
        index: grouped ? shape.group : shape.object,
        kind: shape.kind || "region",
      };
    }

    function setMode(next) {
      mode = next === "pan" || next === "zoom" ? next : "select";
      canvas.classList.remove("mode-select", "mode-pan", "mode-zoom");
      canvas.classList.add(`mode-${mode}`);
      if (hooks && hooks.onMode) hooks.onMode(mode);
    }

    function setSelection(next) {
      selection = next && next.name ? next : null;
      draw();
    }

    function displayKey(display) {
      if (!display) return "";
      const shapes = display.shapes || [];
      return `${display.width}x${display.height}:${(display.image || "").length}:${shapes.length}:${shapes.map((shape) => shape.color || "").join(",")}`;
    }

    async function setDisplay(display) {
      const key = displayKey(display);
      if (key && key === shownKey) return;
      shownKey = key;
      disp = display || null;
      baseImage = null;
      images.clear();
      if (!disp) {
        draw();
        return;
      }
      const jobs = [];
      if (disp.image) jobs.push(loadImg(disp.image).then((image) => { baseImage = image; }).catch(() => {}));
      (disp.shapes || []).forEach((shape) => {
        if (shape.type === "fill" && shape.image && !images.has(shape.image)) {
          images.set(shape.image, null);
          jobs.push(loadImg(shape.image).then((image) => images.set(shape.image, image)).catch(() => {}));
        }
      });
      await Promise.all(jobs);
      const changed = cam.fitW !== disp.width || cam.fitH !== disp.height;
      cam.fitW = disp.width;
      cam.fitH = disp.height;
      if (!cam.user || changed) fit(false);
      else draw();
    }

    canvas.addEventListener("mousedown", (event) => {
      if (event.button !== 0 || !disp) return;
      const point = imagePoint(event);
      if (mode === "zoom") {
        zoomBox = { col0: point.col, row0: point.row, col1: point.col, row1: point.row, sx: point.sx, sy: point.sy };
        return;
      }
      drag = { x: event.clientX, y: event.clientY, camX: cam.x, camY: cam.y, moved: false, point };
    });

    window.addEventListener("mousemove", (event) => {
      if (zoomBox) {
        const point = imagePoint(event);
        zoomBox.col1 = point.col;
        zoomBox.row1 = point.row;
        draw();
        return;
      }
      if (!drag) return;
      const dx = event.clientX - drag.x;
      const dy = event.clientY - drag.y;
      if (Math.hypot(dx, dy) > 3) drag.moved = true;
      if (mode === "pan" && drag.moved) {
        cam.x = drag.camX + dx;
        cam.y = drag.camY + dy;
        cam.user = true;
        draw();
      }
    });

    window.addEventListener("mouseup", (event) => {
      if (zoomBox) {
        const box = zoomBox;
        zoomBox = null;
        const width = Math.abs(box.col1 - box.col0);
        const height = Math.abs(box.row1 - box.row0);
        if (width > 4 && height > 4) zoomToRect(box.col0, box.row0, box.col1, box.row1);
        else zoomBy(1.25);
        return;
      }
      if (!drag) return;
      const info = drag;
      drag = null;
      if (mode === "select" && !info.moved && hooks && hooks.onPick) {
        hooks.onPick(pickAt(info.point.col, info.point.row));
      }
    });

    canvas.addEventListener("wheel", (event) => {
      if (!disp) return;
      event.preventDefault();
      const point = imagePoint(event);
      const next = Math.min(48, Math.max(0.05, cam.scale * (event.deltaY < 0 ? 1.12 : 1 / 1.12)));
      const k = next / cam.scale;
      cam.x = point.sx - (point.sx - cam.x) * k;
      cam.y = point.sy - (point.sy - cam.y) * k;
      cam.scale = next;
      cam.user = true;
      draw();
    }, { passive: false });

    new ResizeObserver(() => resize()).observe(canvas.parentElement || canvas);
    setMode("select");
    resize();
    return { setMode, setDisplay, setSelection, fit, one, zoomBy, resize };
  }

  window.VisdevView = { mount };
})();
