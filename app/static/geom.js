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

  const api = { pointInPoly, lineDistance, pickShape };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevGeom = api;
})();
