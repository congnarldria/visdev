(function () {
  function pointInPoly(x, y, points) {
    let inside = false;
    for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
      const xi = points[i][0];
      const yi = points[i][1];
      const xj = points[j][0];
      const yj = points[j][1];
      const cross = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-9) + xi);
      if (cross) inside = !inside;
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

  function pickShape(shapes, col, row, scale) {
    let regionHit = null;
    let xldHit = null;
    let xldDist = Infinity;
    const limit = 12 / (scale || 1);
    let xldInside = null;
    let xldInsideArea = Infinity;
    const list = shapes || [];
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const shape = list[i];
      if (shape.type !== "polyline" || shape.object == null || !shape.source) continue;
      if (shape.kind === "xld") {
        const points = shape.points || [];
        const dist = lineDistance(col, row, points);
        if (dist <= limit && dist < xldDist) {
          xldDist = dist;
          xldHit = shape;
        }
        const ends = points.length >= 3 ? Math.hypot(points[0][0] - points[points.length - 1][0], points[0][1] - points[points.length - 1][1]) : Infinity;
        if (ends < 1.5 && pointInPoly(col, row, points)) {
          let area = 0;
          for (let p = 0, q = points.length - 1; p < points.length; q = p, p += 1) {
            area += points[q][0] * points[p][1] - points[p][0] * points[q][1];
          }
          area = Math.abs(area);
          if (area < xldInsideArea) {
            xldInsideArea = area;
            xldInside = shape;
          }
        }
      } else if (!regionHit && pointInPoly(col, row, shape.points || [])) {
        regionHit = shape;
      }
    }
    const shape = xldHit || xldInside || regionHit;
    if (!shape) return null;
    const grouped = shape.kind === "xld" && shape.group != null;
    return {
      name: shape.source,
      index: grouped ? shape.group : shape.object,
      kind: shape.kind || "region",
      group: grouped ? shape.group : null,
    };
  }

  function shouldRefit(userAdjusted, prevW, prevH, nextW, nextH) {
    if (!nextW || !nextH) return false;
    if (userAdjusted) return prevW > 0 && prevH > 0 && (prevW !== nextW || prevH !== nextH);
    return true;
  }

  function frameAt(viewW, viewH, scale, col, row) {
    const next = Math.min(48, Math.max(0.05, Number(scale) || 1));
    return {
      scale: next,
      x: viewW / 2 - col * next,
      y: viewH / 2 - row * next,
    };
  }

  function fitView(viewW, viewH, imageW, imageH, pad) {
    const scale = Math.max(0.02, Math.min((viewW - pad * 2) / imageW, (viewH - pad * 2) / imageH));
    return {
      scale,
      x: (viewW - imageW * scale) / 2,
      y: (viewH - imageH * scale) / 2,
    };
  }

  function oneToOne(viewW, viewH, imageW, imageH) {
    return {
      scale: 1,
      x: (viewW - imageW) / 2,
      y: (viewH - imageH) / 2,
    };
  }

  function zoomAt(cam, sx, sy, factor) {
    const next = Math.min(48, Math.max(0.05, cam.scale * factor));
    const k = next / cam.scale;
    return {
      scale: next,
      x: sx - (sx - cam.x) * k,
      y: sy - (sy - cam.y) * k,
    };
  }

  function zoomRect(viewW, viewH, col0, row0, col1, row1, pad) {
    const left = Math.min(col0, col1);
    const top = Math.min(row0, row1);
    const width = Math.max(1, Math.abs(col1 - col0));
    const height = Math.max(1, Math.abs(row1 - row0));
    const scale = Math.min(48, Math.max(0.05, Math.min((viewW - pad * 2) / width, (viewH - pad * 2) / height)));
    return {
      scale,
      x: (viewW - width * scale) / 2 - left * scale,
      y: (viewH - height * scale) / 2 - top * scale,
    };
  }

  function panBy(x, y, dx, dy) {
    return { x: x + dx, y: y + dy };
  }

  function imagePoint(sx, sy, cam) {
    return {
      col: (sx - cam.x) / cam.scale,
      row: (sy - cam.y) / cam.scale,
    };
  }

  function draftBox(a, b) {
    const col1 = Math.min(a.col, b.col);
    const col2 = Math.max(a.col, b.col);
    const row1 = Math.min(a.row, b.row);
    const row2 = Math.max(a.row, b.row);
    return { col1, col2, row1, row2, width: col2 - col1, height: row2 - row1 };
  }

  function geometryTools(names) {
    const has = (name) => (names || []).indexOf(name) !== -1;
    const tools = [];
    if (has("Row1") && has("Column1") && has("Row2") && has("Column2")) tools.push("rect");
    if (has("Row") && has("Column") && has("Length1") && has("Length2") && tools.indexOf("rect") === -1) tools.push("rect");
    if (has("Row") && has("Column") && has("Radius")) tools.push("circle");
    if (has("Row") && has("Column") && has("Radius1") && has("Radius2")) tools.push("ellipse");
    return tools;
  }

  function shapeNumbers(tool, a, b, mode) {
    const box = draftBox(a, b);
    if (tool === "rect" && mode === "corners") {
      return { Row1: box.row1, Column1: box.col1, Row2: box.row2, Column2: box.col2 };
    }
    if (tool === "rect") {
      return {
        Row: (box.row1 + box.row2) / 2,
        Column: (box.col1 + box.col2) / 2,
        Phi: 0,
        Length1: box.width / 2,
        Length2: box.height / 2,
      };
    }
    if (tool === "circle") {
      return {
        Row: a.row,
        Column: a.col,
        Radius: Math.hypot(b.col - a.col, b.row - a.row),
      };
    }
    const along = box.width / 2;
    const across = box.height / 2;
    const inner = mode === "inner";
    return {
      Row: (box.row1 + box.row2) / 2,
      Column: (box.col1 + box.col2) / 2,
      Phi: 0,
      Radius1: inner ? Math.min(along, across) : along,
      Radius2: inner ? Math.max(along, across) : across,
    };
  }

  const api = {
    pointInPoly,
    lineDistance,
    pickShape,
    shouldRefit,
    frameAt,
    fitView,
    oneToOne,
    zoomAt,
    zoomRect,
    panBy,
    imagePoint,
    draftBox,
    geometryTools,
    shapeNumbers,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevGeom = api;
})();
