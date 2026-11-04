namespace Vizion;

/// <summary>次像素邊緣上的一個點。列、行和 Python 的輪廓點相同：輪廓裡是 (行, 列)。</summary>
public readonly record struct EdgePoint(double Row, double Column);

public static partial class Operators
{
    /// <summary>高斯導數的梯度在法線上做拋物線內插，邊緣落在強度最高的次像素，而不是整數像素的外框。Low 以上才相連，其中要有一點達到 High。黑白階躍在 Sigma 1 時強度大約 60，門檻 20 和 40 留得住明顯的邊。差幾個灰階的雜訊不會留下。彩色先轉成亮度。Alpha 要大於 0，High 不能小於 Low。</summary>
    /// <remarks>萃取次像素邊緣</remarks>
    /// <example>
    /// <code>Vizion.edges_sub_pix (Image, Edges, 'canny', 1.0, 20, 40)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="alpha">平滑標準差，要大於 0。</param>
    /// <param name="low">梯度低門檻。</param>
    /// <param name="high">能當種子的高門檻，不能小於 Low。</param>
    /// <returns>次像素邊緣。這版只做 canny。</returns>
    public static EdgePoint[][] EdgesSubPix(ImageBuffer image, double alpha, double low, double high)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (alpha <= 0)
            throw new ArgumentOutOfRangeException(nameof(alpha), "Alpha 要大於 0");
        if (high < low)
            throw new ArgumentOutOfRangeException(nameof(high), "High 不能小於 Low");

        int width = image.Width;
        int height = image.Height;
        var gray = new float[width * height];
        byte[] source = image.GrayBytes();
        for (int i = 0; i < gray.Length; i++)
            gray[i] = source[i];

        Kernels(alpha, out float[] smooth, out float[] first);
        var magnitude = new float[gray.Length];
        var normalX = new float[gray.Length];
        var normalY = new float[gray.Length];
        var shift = new float[gray.Length];
        var center = new bool[gray.Length];
        bool[]? domain = DomainMask(image);
        var nativeDx = new float[gray.Length];
        var nativeDy = new float[gray.Length];
        var nativeCenter = new byte[gray.Length];
        if (NativeCore.TrySubpix(gray, width, height, smooth, first, nativeDx, nativeDy, magnitude, normalX, normalY, shift, nativeCenter))
        {
            for (int i = 0; i < gray.Length; i++)
            {
                bool on = nativeCenter[i] != 0;
                if (domain != null)
                    on &= domain[i];
                center[i] = on;
            }
        }
        else
        {
            float[] dx = Sep(gray, width, height, first, smooth);
            float[] dy = Sep(gray, width, height, smooth, first);
            for (int i = 0; i < gray.Length; i++)
            {
                float mag = (float)Math.Sqrt(dx[i] * dx[i] + dy[i] * dy[i]);
                magnitude[i] = mag;
                float scale = Math.Max(mag, 1e-6f);
                normalX[i] = dx[i] / scale;
                normalY[i] = dy[i] / scale;
            }
            for (int row = 0; row < height; row++)
            {
                for (int col = 0; col < width; col++)
                {
                    int index = row * width + col;
                    float ahead = Sample(magnitude, width, height, row + normalY[index], col + normalX[index]);
                    float behind = Sample(magnitude, width, height, row - normalY[index], col - normalX[index]);
                    float denom = behind + ahead - 2f * magnitude[index];
                    float moved = Math.Abs(denom) > 1e-6f ? 0.5f * (behind - ahead) / denom : 9f;
                    shift[index] = moved;
                    // Python 用 OpenCV 的 float32，邊界上的位移會剛好是 -0.5，所以不算中心。
                    // 這裡的卷積可能差 1 ulp，-0.5 會變成 -0.49999997。1e-6 把這個邊界留在同一側。
                    bool on = magnitude[index] >= ahead && magnitude[index] >= behind && magnitude[index] > 0
                        && moved > -0.5f + 1e-6f && moved <= 0.5f;
                    if (domain != null)
                        on &= domain[index];
                    center[index] = on;
                }
            }
        }

        var centerPacked = new byte[gray.Length];
        for (int i = 0; i < gray.Length; i++)
            centerPacked[i] = (byte)(center[i] ? 1 : 0);
        if (NativeCore.TryEdgeChains(centerPacked, magnitude, width, height, low, high, shift, normalX, normalY, out EdgePoint[][] linked))
            return linked;

        var keep = new bool[gray.Length];
        var seen = new bool[gray.Length];
        var stack = new Stack<int>();
        for (int i = 0; i < gray.Length; i++)
        {
            if (!center[i] || magnitude[i] < high || seen[i])
                continue;
            stack.Push(i);
            seen[i] = true;
            while (stack.Count > 0)
            {
                int current = stack.Pop();
                keep[current] = true;
                int row = current / width;
                int col = current - row * width;
                for (int dRow = -1; dRow <= 1; dRow++)
                {
                    int nextRow = row + dRow;
                    if (nextRow < 0 || nextRow >= height)
                        continue;
                    for (int dCol = -1; dCol <= 1; dCol++)
                    {
                        if (dRow == 0 && dCol == 0)
                            continue;
                        int nextCol = col + dCol;
                        if (nextCol < 0 || nextCol >= width)
                            continue;
                        int next = nextRow * width + nextCol;
                        if (seen[next] || !center[next] || magnitude[next] < low)
                            continue;
                        seen[next] = true;
                        stack.Push(next);
                    }
                }
            }
        }

        return Link(keep, width, height, shift, normalX, normalY);
    }

    static EdgePoint[][] Link(bool[] mask, int width, int height, float[] shift, float[] normalX, float[] normalY)
    {
        var points = new List<(int Row, int Col)>();
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                if (mask[row * width + col])
                    points.Add((row, col));
            }
        }
        if (points.Count == 0)
            return Array.Empty<EdgePoint[]>();

        var set = new HashSet<(int Row, int Col)>(points);
        var neighbors = new Dictionary<(int Row, int Col), List<(int Row, int Col)>>();
        foreach (var point in points)
        {
            var found = new List<(int Row, int Col)>();
            for (int dRow = -1; dRow <= 1; dRow++)
            {
                for (int dCol = -1; dCol <= 1; dCol++)
                {
                    if (dRow == 0 && dCol == 0)
                        continue;
                    var next = (point.Row + dRow, point.Col + dCol);
                    if (set.Contains(next))
                        found.Add(next);
                }
            }
            neighbors[point] = found;
        }

        var used = new HashSet<((int Row, int Col) A, (int Row, int Col) B)>();
        var chains = new List<List<(int Row, int Col)>>();

        void Walk((int Row, int Col) start, (int Row, int Col) next)
        {
            var key = Edge(start, next);
            if (!used.Add(key))
                return;
            var chain = new List<(int Row, int Col)> { start };
            var current = next;
            while (true)
            {
                chain.Add(current);
                var choices = new List<(int Row, int Col)>();
                foreach (var item in neighbors[current])
                {
                    if (!used.Contains(Edge(current, item)))
                        choices.Add(item);
                }
                if (choices.Count != 1)
                    break;
                var previous = current;
                current = choices[0];
                used.Add(Edge(previous, current));
            }
            if (chain.Count >= 2)
                chains.Add(chain);
        }

        var starts = new List<(int Row, int Col)>();
        foreach (var pair in neighbors)
        {
            if (pair.Value.Count != 2)
                starts.Add(pair.Key);
        }
        if (starts.Count == 0)
            starts.Add(points.Min());
        starts.Sort();
        foreach (var start in starts)
        {
            foreach (var next in neighbors[start])
                Walk(start, next);
        }

        var lines = new EdgePoint[chains.Count][];
        for (int i = 0; i < chains.Count; i++)
        {
            var chain = chains[i];
            var line = new EdgePoint[chain.Count];
            for (int k = 0; k < chain.Count; k++)
            {
                int row = chain[k].Row;
                int col = chain[k].Col;
                int index = row * width + col;
                float moved = shift[index];
                line[k] = new EdgePoint(row + moved * normalY[index], col + moved * normalX[index]);
            }
            lines[i] = line;
        }
        return lines;
    }

    static ((int Row, int Col) A, (int Row, int Col) B) Edge((int Row, int Col) left, (int Row, int Col) right)
    {
        if (left.Row < right.Row || (left.Row == right.Row && left.Col <= right.Col))
            return (left, right);
        return (right, left);
    }

    static void Kernels(double sigma, out float[] smooth, out float[] first)
    {
        int radius = Math.Max(1, (int)Math.Ceiling(sigma * 4.0));
        int length = radius * 2 + 1;
        var smooth64 = new double[length];
        var first64 = new double[length];
        double variance = sigma * sigma;
        double sum = 0;
        for (int i = 0; i < length; i++)
        {
            double axis = i - radius;
            smooth64[i] = Math.Exp(-0.5 * axis * axis / variance);
            sum += smooth64[i];
        }
        for (int i = 0; i < length; i++)
        {
            smooth64[i] /= sum;
            double axis = i - radius;
            first64[i] = (-axis / variance) * smooth64[i];
        }
        smooth = new float[length];
        first = new float[length];
        for (int i = 0; i < length; i++)
        {
            smooth[i] = (float)smooth64[i];
            first[i] = (float)first64[i];
        }
    }

    static float[] Sep(float[] source, int width, int height, float[] across, float[] along)
    {
        int radiusX = across.Length / 2;
        int radiusY = along.Length / 2;
        var horizontal = new float[source.Length];
        var destination = new float[source.Length];
        for (int row = 0; row < height; row++)
        {
            int offset = row * width;
            for (int col = 0; col < width; col++)
            {
                float sum = 0;
                for (int k = 0; k < across.Length; k++)
                {
                    int xx = Reflect101(col + k - radiusX, width);
                    sum += source[offset + xx] * across[k];
                }
                horizontal[offset + col] = sum;
            }
        }
        for (int row = 0; row < height; row++)
        {
            for (int col = 0; col < width; col++)
            {
                float sum = 0;
                for (int k = 0; k < along.Length; k++)
                {
                    int yy = Reflect101(row + k - radiusY, height);
                    sum += horizontal[yy * width + col] * along[k];
                }
                destination[row * width + col] = sum;
            }
        }
        return destination;
    }

    static int Reflect101(int position, int length)
    {
        if ((uint)position < (uint)length)
            return position;
        if (length == 1)
            return 0;
        do
        {
            if (position < 0)
                position = -position - 1;
            else
                position = length - 1 - (position - length);
        }
        while ((uint)position >= (uint)length);
        return position;
    }

    static float Sample(float[] field, int width, int height, float row, float column)
    {
        if (column < 0) column = 0;
        if (row < 0) row = 0;
        if (column > width - 1) column = width - 1;
        if (row > height - 1) row = height - 1;
        int x0 = (int)Math.Floor(column);
        int y0 = (int)Math.Floor(row);
        int x1 = Math.Min(x0 + 1, width - 1);
        int y1 = Math.Min(y0 + 1, height - 1);
        float wx = column - x0;
        float wy = row - y0;
        float top = (1 - wx) * field[y0 * width + x0] + wx * field[y0 * width + x1];
        float bottom = (1 - wx) * field[y1 * width + x0] + wx * field[y1 * width + x1];
        return (1 - wy) * top + wy * bottom;
    }

    static bool[]? DomainMask(ImageBuffer image)
    {
        if (image.Domain == null)
            return null;
        var mask = new bool[image.Width * image.Height];
        var domain = image.Domain;
        for (int i = 0; i < domain.Rows.Length; i++)
        {
            int row = domain.Rows[i];
            int column = domain.Columns[i];
            int length = domain.Lengths[i];
            for (int x = 0; x < length; x++)
                mask[row * image.Width + column + x] = true;
        }
        return mask;
    }
}
