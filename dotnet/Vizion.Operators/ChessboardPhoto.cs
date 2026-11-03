namespace Vizion;

/// <summary>
/// 把棋盤畫成一張灰階圖，再從圖上取出內角點。
/// 角點順序和 CalibPlate.Point(row, column) 相同：先沿行，再沿列。
/// 不找六角形標記，也不改 Python 的角點或九個參數。
/// </summary>
public static class ChessboardPhoto
{
    public static ImageBuffer Render(
        int width,
        int height,
        Intrinsics camera,
        double[,] rotation,
        double tx,
        double ty,
        double tz,
        int innerRows,
        int innerColumns,
        double square)
    {
        if (width < 8 || height < 8)
            throw new ArgumentOutOfRangeException(nameof(width));
        if (innerRows < 2 || innerColumns < 2 || square <= 0)
            throw new ArgumentOutOfRangeException(nameof(innerRows));
        var pixels = new byte[width * height];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = 128;
        (double X, double Y) Project(double x, double y)
        {
            double cameraX = rotation[0, 0] * x + rotation[0, 1] * y + rotation[0, 2] * 0 + tx;
            double cameraY = rotation[1, 0] * x + rotation[1, 1] * y + rotation[1, 2] * 0 + ty;
            double cameraZ = rotation[2, 0] * x + rotation[2, 1] * y + rotation[2, 2] * 0 + tz;
            if (cameraZ < 1)
                return (double.NaN, double.NaN);
            return (camera.Fx * cameraX / cameraZ + camera.Cx, camera.Fy * cameraY / cameraZ + camera.Cy);
        }
        for (int iy = -1; iy < innerRows; iy++)
        {
            for (int ix = -1; ix < innerColumns; ix++)
            {
                double x0 = ix * square;
                double y0 = iy * square;
                var corners = new (double X, double Y)[]
                {
                    Project(x0, y0),
                    Project(x0 + square, y0),
                    Project(x0 + square, y0 + square),
                    Project(x0, y0 + square),
                };
                if (corners.Any(item => double.IsNaN(item.X)))
                    continue;
                byte color = ((ix + iy) & 1) == 0 ? (byte)255 : (byte)0;
                FillConvex(pixels, width, height, corners, color);
            }
        }
        return ImageBuffer.Gray(width, height, pixels);
    }

    public static PlanarCalib.View FindCorners(ImageBuffer image, int innerRows, int innerColumns)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (innerRows < 2 || innerColumns < 2)
            throw new ArgumentOutOfRangeException(nameof(innerRows));
        byte[] gray = image.GrayBytes();
        int width = image.Width;
        int height = image.Height;
        var labels = new int[width * height];
        int count = 0;
        var stack = new Stack<int>();
        for (int index = 0; index < gray.Length; index++)
        {
            if (gray[index] >= 80 || labels[index] != 0)
                continue;
            count++;
            labels[index] = count;
            stack.Push(index);
            while (stack.Count > 0)
            {
                int pixel = stack.Pop();
                int row = pixel / width;
                int column = pixel - row * width;
                void Push(int nextRow, int nextColumn)
                {
                    if ((uint)nextRow >= (uint)height || (uint)nextColumn >= (uint)width)
                        return;
                    int next = nextRow * width + nextColumn;
                    if (labels[next] != 0 || gray[next] >= 80)
                        return;
                    labels[next] = count;
                    stack.Push(next);
                }
                Push(row - 1, column);
                Push(row + 1, column);
                Push(row, column - 1);
                Push(row, column + 1);
            }
        }
        var corners = new List<(double X, double Y)>();
        for (int label = 1; label <= count; label++)
        {
            var pixels = new List<(double X, double Y)>();
            for (int index = 0; index < labels.Length; index++)
            {
                if (labels[index] != label)
                    continue;
                int row = index / width;
                pixels.Add((index - row * width + 0.5, row + 0.5));
            }
            if (pixels.Count < 40)
                continue;
            foreach (var corner in QuadCorners(pixels))
                corners.Add(corner);
        }
        var merged = Cluster(corners, 4);
        var shared = merged.Where(item => item.Count >= 2).Select(item => Refine(gray, width, height, item.X, item.Y)).ToList();
        int expect = innerRows * innerColumns;
        if (shared.Count != expect)
            throw new InvalidOperationException($"棋盤應有 {expect} 個內角點，圖上得到 {shared.Count} 個");
        return Order(shared, innerRows, innerColumns);
    }

    static void FillConvex(byte[] pixels, int width, int height, (double X, double Y)[] corners, byte color)
    {
        double minY = corners.Min(item => item.Y);
        double maxY = corners.Max(item => item.Y);
        int y0 = Math.Max(0, (int)Math.Floor(minY));
        int y1 = Math.Min(height - 1, (int)Math.Ceiling(maxY));
        for (int row = y0; row <= y1; row++)
        {
            double scan = row + 0.5;
            double left = double.PositiveInfinity;
            double right = double.NegativeInfinity;
            int hits = 0;
            for (int i = 0; i < corners.Length; i++)
            {
                var a = corners[i];
                var b = corners[(i + 1) % corners.Length];
                if (Math.Abs(a.Y - b.Y) < 1e-9)
                    continue;
                if ((a.Y <= scan && b.Y > scan) || (b.Y <= scan && a.Y > scan))
                {
                    double t = (scan - a.Y) / (b.Y - a.Y);
                    double x = a.X + t * (b.X - a.X);
                    if (x < left)
                        left = x;
                    if (x > right)
                        right = x;
                    hits++;
                }
            }
            if (hits < 2)
                continue;
            int x0 = Math.Max(0, (int)Math.Floor(left));
            int x1 = Math.Min(width - 1, (int)Math.Floor(right));
            for (int column = x0; column <= x1; column++)
            {
                if (column + 0.5 < left || column + 0.5 > right)
                    continue;
                pixels[row * width + column] = color;
            }
        }
    }

    static (double X, double Y)[] QuadCorners(List<(double X, double Y)> pixels)
    {
        var hull = Hull(pixels);
        if (hull.Count < 4)
            return Array.Empty<(double X, double Y)>();
        double cx = 0, cy = 0;
        foreach (var point in pixels)
        {
            cx += point.X;
            cy += point.Y;
        }
        cx /= pixels.Count;
        cy /= pixels.Count;
        var turns = new List<(int Index, double Angle)>();
        for (int i = 0; i < hull.Count; i++)
        {
            var previous = hull[(i - 1 + hull.Count) % hull.Count];
            var current = hull[i];
            var next = hull[(i + 1) % hull.Count];
            double ax = current.X - previous.X;
            double ay = current.Y - previous.Y;
            double bx = next.X - current.X;
            double by = next.Y - current.Y;
            double scale = Math.Sqrt((ax * ax + ay * ay) * (bx * bx + by * by));
            if (scale < 1e-9)
                continue;
            double turn = Math.Abs(Math.Atan2(ax * by - ay * bx, ax * bx + ay * by));
            turns.Add((i, turn));
        }
        var chosen = turns.OrderByDescending(item => item.Angle).Take(4).Select(item => item.Index).OrderBy(item => item).ToList();
        if (chosen.Count < 4)
            return Array.Empty<(double X, double Y)>();
        var lines = new (double X, double Y, double Dx, double Dy)[4];
        for (int side = 0; side < 4; side++)
        {
            int begin = chosen[side];
            int end = chosen[(side + 1) % 4];
            var samples = new List<(double X, double Y)>();
            for (int i = begin; i != end; i = (i + 1) % hull.Count)
                samples.Add(hull[i]);
            samples.Add(hull[end]);
            double mx = 0, my = 0;
            foreach (var sample in samples)
            {
                mx += sample.X;
                my += sample.Y;
            }
            mx /= samples.Count;
            my /= samples.Count;
            double sxx = 0, syy = 0, sxy = 0;
            foreach (var sample in samples)
            {
                double dx = sample.X - mx;
                double dy = sample.Y - my;
                sxx += dx * dx;
                syy += dy * dy;
                sxy += dx * dy;
            }
            double angle = 0.5 * Math.Atan2(2 * sxy, sxx - syy);
            double dxDir = Math.Cos(angle);
            double dyDir = Math.Sin(angle);
            double outwardX = mx - cx;
            double outwardY = my - cy;
            double norm = Math.Sqrt(outwardX * outwardX + outwardY * outwardY);
            if (norm > 1e-9)
            {
                mx += outwardX / norm * 0.5;
                my += outwardY / norm * 0.5;
            }
            lines[side] = (mx, my, dxDir, dyDir);
        }
        var corners = new (double X, double Y)[4];
        for (int side = 0; side < 4; side++)
        {
            var a = lines[side];
            var b = lines[(side + 1) % 4];
            double det = a.Dx * b.Dy - a.Dy * b.Dx;
            if (Math.Abs(det) < 1e-9)
                return Array.Empty<(double X, double Y)>();
            double t = ((b.X - a.X) * b.Dy - (b.Y - a.Y) * b.Dx) / det;
            corners[side] = (a.X + t * a.Dx, a.Y + t * a.Dy);
        }
        return corners;
    }

    static List<(double X, double Y)> Hull(List<(double X, double Y)> points)
    {
        var unique = points.Distinct().OrderBy(item => item.X).ThenBy(item => item.Y).ToList();
        double Cross((double X, double Y) origin, (double X, double Y) a, (double X, double Y) b) =>
            (a.X - origin.X) * (b.Y - origin.Y) - (a.Y - origin.Y) * (b.X - origin.X);
        var lower = new List<(double X, double Y)>();
        foreach (var point in unique)
        {
            while (lower.Count >= 2 && Cross(lower[lower.Count - 2], lower[lower.Count - 1], point) <= 0)
                lower.RemoveAt(lower.Count - 1);
            lower.Add(point);
        }
        var upper = new List<(double X, double Y)>();
        for (int i = unique.Count - 1; i >= 0; i--)
        {
            var point = unique[i];
            while (upper.Count >= 2 && Cross(upper[upper.Count - 2], upper[upper.Count - 1], point) <= 0)
                upper.RemoveAt(upper.Count - 1);
            upper.Add(point);
        }
        if (lower.Count > 0)
            lower.RemoveAt(lower.Count - 1);
        if (upper.Count > 0)
            upper.RemoveAt(upper.Count - 1);
        lower.AddRange(upper);
        return lower;
    }

    readonly record struct ClusterHit(double X, double Y, int Count)
    {
        public (double X, double Y) At => (X, Y);
    }

    static List<ClusterHit> Cluster(List<(double X, double Y)> corners, double radius)
    {
        var used = new bool[corners.Count];
        var groups = new List<ClusterHit>();
        double limit = radius * radius;
        for (int i = 0; i < corners.Count; i++)
        {
            if (used[i])
                continue;
            double sx = 0, sy = 0;
            int count = 0;
            for (int j = i; j < corners.Count; j++)
            {
                if (used[j])
                    continue;
                double dx = corners[j].X - corners[i].X;
                double dy = corners[j].Y - corners[i].Y;
                if (i != j && dx * dx + dy * dy > limit)
                    continue;
                used[j] = true;
                sx += corners[j].X;
                sy += corners[j].Y;
                count++;
            }
            groups.Add(new ClusterHit(sx / count, sy / count, count));
        }
        return groups;
    }

    static (double X, double Y) Refine(byte[] gray, int width, int height, double x, double y)
    {
        int bestRadius = 8;
        double bestScore = -1;
        foreach (int radius in new[] { 6, 10, 14 })
        {
            double score = Checker(gray, width, height, x, y, radius);
            if (score > bestScore)
            {
                bestScore = score;
                bestRadius = radius;
            }
        }
        double seedX = x;
        double seedY = y;
        for (double dy = -3; dy <= 3.001; dy += 0.5)
        {
            for (double dx = -3; dx <= 3.001; dx += 0.5)
            {
                double score = Checker(gray, width, height, seedX + dx, seedY + dy, bestRadius);
                if (score > bestScore)
                {
                    bestScore = score;
                    x = seedX + dx;
                    y = seedY + dy;
                }
            }
        }
        for (int pass = 0; pass < 2; pass++)
        {
            double center = Checker(gray, width, height, x, y, bestRadius);
            double shiftX = PeakShift(
                Checker(gray, width, height, x - 0.5, y, bestRadius),
                center,
                Checker(gray, width, height, x + 0.5, y, bestRadius));
            double shiftY = PeakShift(
                Checker(gray, width, height, x, y - 0.5, bestRadius),
                center,
                Checker(gray, width, height, x, y + 0.5, bestRadius));
            x += shiftX * 0.5;
            y += shiftY * 0.5;
        }
        return (x, y);
    }

    static double PeakShift(double left, double center, double right)
    {
        double denom = left - 2 * center + right;
        if (Math.Abs(denom) < 1e-6 || center < left || center < right)
            return 0;
        double shift = 0.5 * (left - right) / denom;
        if (shift < -0.75)
            return -0.75;
        if (shift > 0.75)
            return 0.75;
        return shift;
    }

    static double Checker(byte[] gray, int width, int height, double x, double y, int radius)
    {
        double At(double column, double row)
        {
            if (column < 0 || row < 0 || column > width - 1 || row > height - 1)
                return 128;
            int x0 = (int)Math.Floor(column);
            int y0 = (int)Math.Floor(row);
            int x1 = Math.Min(width - 1, x0 + 1);
            int y1 = Math.Min(height - 1, y0 + 1);
            double wx = column - x0;
            double wy = row - y0;
            double top = gray[y0 * width + x0] * (1 - wx) + gray[y0 * width + x1] * wx;
            double bottom = gray[y1 * width + x0] * (1 - wx) + gray[y1 * width + x1] * wx;
            return top * (1 - wy) + bottom * wy;
        }
        double northwest = At(x - radius, y - radius);
        double southeast = At(x + radius, y + radius);
        double northeast = At(x + radius, y - radius);
        double southwest = At(x - radius, y + radius);
        double contrast = Math.Abs(northwest + southeast - northeast - southwest);
        double balance = Math.Abs(northwest - southeast) + Math.Abs(northeast - southwest);
        if (contrast < balance)
            return 0;
        return contrast;
    }

    static PlanarCalib.View Order(List<(double X, double Y)> points, int rows, int columns)
    {
        var nearest = new double[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            double best = double.PositiveInfinity;
            for (int j = 0; j < points.Count; j++)
            {
                if (i == j)
                    continue;
                double dx = points[i].X - points[j].X;
                double dy = points[i].Y - points[j].Y;
                double distance = dx * dx + dy * dy;
                if (distance < best)
                    best = distance;
            }
            nearest[i] = Math.Sqrt(best);
        }
        Array.Sort(nearest);
        double step = nearest[nearest.Length / 2];
        double low = step * 0.45;
        double high = step * 2.8;
        var neighbors = new List<int>[points.Count];
        for (int i = 0; i < points.Count; i++)
        {
            neighbors[i] = new List<int>();
            for (int j = 0; j < points.Count; j++)
            {
                if (i == j)
                    continue;
                double dx = points[i].X - points[j].X;
                double dy = points[i].Y - points[j].Y;
                double distance = Math.Sqrt(dx * dx + dy * dy);
                if (distance >= low && distance <= high)
                    neighbors[i].Add(j);
            }
            neighbors[i] = neighbors[i].OrderBy(index =>
            {
                double dx = points[i].X - points[index].X;
                double dy = points[i].Y - points[index].Y;
                return dx * dx + dy * dy;
            }).ToList();
        }
        var hull = Hull(points);
        if (hull.Count < 4)
            throw new InvalidOperationException("棋盤角點連不成格子");
        var turns = new List<(int Index, double Angle)>();
        for (int i = 0; i < hull.Count; i++)
        {
            var previous = hull[(i - 1 + hull.Count) % hull.Count];
            var current = hull[i];
            var next = hull[(i + 1) % hull.Count];
            double ax = current.X - previous.X;
            double ay = current.Y - previous.Y;
            double bx = next.X - current.X;
            double by = next.Y - current.Y;
            double scale = Math.Sqrt((ax * ax + ay * ay) * (bx * bx + by * by));
            if (scale < 1e-9)
                continue;
            turns.Add((i, Math.Abs(Math.Atan2(ax * by - ay * bx, ax * bx + ay * by))));
        }
        var cornerIndexes = turns.OrderByDescending(item => item.Angle).Take(4).Select(item => item.Index).OrderBy(item => item).ToList();
        if (cornerIndexes.Count < 4)
            throw new InvalidOperationException("棋盤角點連不成格子");
        int originHull = cornerIndexes
            .OrderBy(index => hull[index].Y)
            .ThenBy(index => hull[index].X)
            .First();
        int origin = NearestIndex(points, hull[originHull]);
        var otherCorners = cornerIndexes.Where(index => index != originHull)
            .Select(index => NearestIndex(points, hull[index]))
            .OrderBy(index =>
            {
                double dx = points[index].X - points[origin].X;
                double dy = points[index].Y - points[origin].Y;
                return dx * dx + dy * dy;
            }).ToList();
        var alongFirst = WalkToward(points, origin, otherCorners[0], step);
        var alongSecond = WalkToward(points, origin, otherCorners[1], step);
        bool firstIsColumn = alongFirst.Count >= alongSecond.Count;
        var alongColumn = firstIsColumn ? alongFirst : alongSecond;
        var alongRow = firstIsColumn ? alongSecond : alongFirst;
        if (alongColumn.Count != columns || alongRow.Count != rows)
            throw new InvalidOperationException($"棋盤邊長應是 {columns} 和 {rows}，走到 {alongColumn.Count} 和 {alongRow.Count}");
        int opposite = otherCorners[2];
        var rightSide = WalkToward(points, alongColumn[alongColumn.Count - 1], opposite, step);
        if (rightSide.Count != rows)
            throw new InvalidOperationException($"棋盤右側應有 {rows} 個點，走到 {rightSide.Count}");
        var grid = new int[rows, columns];
        for (int c = 0; c < columns; c++)
            grid[0, c] = alongColumn[c];
        for (int r = 0; r < rows; r++)
        {
            var across = r == 0 ? alongColumn : WalkToward(points, alongRow[r], rightSide[r], step);
            if (across.Count != columns)
                throw new InvalidOperationException($"棋盤第 {r} 列應有 {columns} 個點，走到 {across.Count}");
            for (int c = 0; c < columns; c++)
                grid[r, c] = across[c];
        }
        var column = new double[rows * columns];
        var row = new double[rows * columns];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                column[r * columns + c] = points[grid[r, c]].X;
                row[r * columns + c] = points[grid[r, c]].Y;
            }
        }
        return new PlanarCalib.View(column, row);
    }

    static int NearestIndex(List<(double X, double Y)> points, (double X, double Y) target)
    {
        int nearest = 0;
        double best = double.PositiveInfinity;
        for (int i = 0; i < points.Count; i++)
        {
            double dx = points[i].X - target.X;
            double dy = points[i].Y - target.Y;
            double distance = dx * dx + dy * dy;
            if (distance < best)
            {
                best = distance;
                nearest = i;
            }
        }
        return nearest;
    }

    static List<int> WalkToward(List<(double X, double Y)> points, int start, int target, double step)
    {
        var chain = new List<int> { start };
        int current = start;
        var seen = new HashSet<int> { start };
        for (int guard = 0; guard < points.Count && current != target; guard++)
        {
            double dx = points[target].X - points[current].X;
            double dy = points[target].Y - points[current].Y;
            double direction = Math.Sqrt(dx * dx + dy * dy);
            int best = -1;
            double bestAlong = double.PositiveInfinity;
            for (int candidate = 0; candidate < points.Count; candidate++)
            {
                if (seen.Contains(candidate))
                    continue;
                double cx = points[candidate].X - points[current].X;
                double cy = points[candidate].Y - points[current].Y;
                double along = direction < 1e-9 ? 0 : (dx * cx + dy * cy) / direction;
                if (along <= step * 0.3)
                    continue;
                double side = Math.Abs(dx * cy - dy * cx) / Math.Max(direction, 1e-9);
                if (side > step * 0.45)
                    continue;
                if (along < bestAlong)
                {
                    bestAlong = along;
                    best = candidate;
                }
            }
            if (best < 0)
                break;
            seen.Add(best);
            chain.Add(best);
            current = best;
        }
        return chain;
    }

    static (double X, double Y) Delta(List<(double X, double Y)> points, int from, int to)
    {
        double dx = points[to].X - points[from].X;
        double dy = points[to].Y - points[from].Y;
        double length = Math.Sqrt(dx * dx + dy * dy);
        return length < 1e-9 ? (0, 0) : (dx, dy);
    }

    static List<int> Walk(List<(double X, double Y)> points, List<int>[] neighbors, int start, int next, double high)
    {
        var chain = new List<int> { start, next };
        int previous = start;
        int current = next;
        for (int guard = 0; guard < points.Count; guard++)
        {
            double dx = points[current].X - points[previous].X;
            double dy = points[current].Y - points[previous].Y;
            int best = -1;
            double bestDot = 0.5;
            foreach (int candidate in neighbors[current])
            {
                if (candidate == previous)
                    continue;
                double cx = points[candidate].X - points[current].X;
                double cy = points[candidate].Y - points[current].Y;
                double scale = Math.Sqrt((dx * dx + dy * dy) * (cx * cx + cy * cy));
                if (scale < 1e-9)
                    continue;
                double dot = (dx * cx + dy * cy) / scale;
                if (dot > bestDot)
                {
                    bestDot = dot;
                    best = candidate;
                }
            }
            if (best < 0 || chain.Contains(best))
                break;
            chain.Add(best);
            previous = current;
            current = best;
        }
        return chain;
    }
}
