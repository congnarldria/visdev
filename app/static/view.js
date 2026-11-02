(function () {
  function loadImg(url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("image"));
      image.src = url;
    });
  }

  function mix(hash, text) {
    const value = String(text);
    let next = hash;
    for (let i = 0; i < value.length; i += 1) next = (Math.imul(next, 33) ^ value.charCodeAt(i)) >>> 0;
    return next;
  }

  function displayKey(display) {
    if (!display) return "";
    let hash = 5381;
    hash = mix(hash, display.width);
    hash = mix(hash, "x");
    hash = mix(hash, display.height);
    hash = mix(hash, display.image || "");
    for (const shape of display.shapes || []) {
      hash = mix(hash, "|");
      hash = mix(hash, shape.type || "");
      hash = mix(hash, shape.color || "");
      hash = mix(hash, shape.text || "");
      hash = mix(hash, shape.row == null ? "" : shape.row);
      hash = mix(hash, shape.col == null ? "" : shape.col);
      hash = mix(hash, shape.image || "");
      const points = shape.points || [];
      hash = mix(hash, points.length);
      for (let i = 0; i < points.length; i += 1) {
        hash = mix(hash, points[i][0]);
        hash = mix(hash, ",");
        hash = mix(hash, points[i][1]);
        hash = mix(hash, ";");
      }
    }
    return String(hash);
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
      const sx = event.clientX - rect.left;
      const sy = event.clientY - rect.top;
      const image = VisdevGeom.imagePoint(sx, sy, cam);
      return { sx, sy, col: image.col, row: image.row };
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
      const placed = VisdevGeom.fitView(rect.width, rect.height, disp.width, disp.height, 12);
      cam.scale = placed.scale;
      cam.x = placed.x;
      cam.y = placed.y;
      cam.user = !!markUser;
      draw();
    }

    function one() {
      const rect = canvas.getBoundingClientRect();
      if (!disp) return;
      const placed = VisdevGeom.oneToOne(rect.width, rect.height, disp.width, disp.height);
      cam.scale = placed.scale;
      cam.x = placed.x;
      cam.y = placed.y;
      cam.user = true;
      draw();
    }

    function zoomBy(factor) {
      const rect = canvas.getBoundingClientRect();
      const placed = VisdevGeom.zoomAt(cam, rect.width / 2, rect.height / 2, factor);
      cam.scale = placed.scale;
      cam.x = placed.x;
      cam.y = placed.y;
      cam.user = true;
      draw();
    }

    function zoomToRect(col0, row0, col1, row1) {
      const rect = canvas.getBoundingClientRect();
      const placed = VisdevGeom.zoomRect(rect.width, rect.height, col0, row0, col1, row1, 8);
      cam.scale = placed.scale;
      cam.x = placed.x;
      cam.y = placed.y;
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
        } else if (shape.type === "text") {
          const size = 14 / cam.scale;
          ctx.font = `${size}px "Segoe UI", "Microsoft JhengHei", sans-serif`;
          ctx.lineWidth = 3 / cam.scale;
          ctx.strokeStyle = "rgba(0,0,0,0.75)";
          ctx.strokeText(shape.text || "", shape.col || 0, shape.row || 0);
          ctx.fillStyle = shape.color || "#fff";
          ctx.fillText(shape.text || "", shape.col || 0, shape.row || 0);
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
      const hit = VisdevGeom.pickShape(disp && disp.shapes || [], col, row, cam.scale);
      if (!hit) return null;
      return { name: hit.name, index: hit.index, kind: hit.kind };
    }

    function setMode(next) {
      mode = next === "pan" || next === "zoom" ? next : "select";
      canvas.classList.remove("mode-select", "mode-pan", "mode-zoom");
      canvas.classList.add(`mode-${mode}`);
      if (hooks && hooks.onMode) hooks.onMode(mode);
    }

    function setFrame(frame) {
      if (!frame || cam.user) return false;
      const rect = canvas.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2) return false;
      const placed = VisdevGeom.frameAt(rect.width, rect.height, frame.scale, frame.col, frame.row);
      cam.scale = placed.scale;
      cam.x = placed.x;
      cam.y = placed.y;
      cam.user = true;
      draw();
      return true;
    }

    function setSelection(next) {
      selection = next && next.name ? next : null;
      draw();
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
      const refit = VisdevGeom.shouldRefit(cam.user, cam.fitW, cam.fitH, disp.width, disp.height);
      cam.fitW = disp.width;
      cam.fitH = disp.height;
      if (refit) fit(false);
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
        const shifted = VisdevGeom.panBy(drag.camX, drag.camY, dx, dy);
        cam.x = shifted.x;
        cam.y = shifted.y;
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
      const placed = VisdevGeom.zoomAt(cam, point.sx, point.sy, event.deltaY < 0 ? 1.12 : 1 / 1.12);
      cam.x = placed.x;
      cam.y = placed.y;
      cam.scale = placed.scale;
      cam.user = true;
      draw();
    }, { passive: false });

    new ResizeObserver(() => resize()).observe(canvas.parentElement || canvas);
    setMode("select");
    resize();
    return { setMode, setDisplay, setSelection, setFrame, fit, one, zoomBy, resize };
  }

  window.VisdevView = { mount, displayKey };
})();
