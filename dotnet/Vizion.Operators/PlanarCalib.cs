namespace Vizion;

/// <summary>
/// 從至少三個視角的平面對應點估計 fx、fy、cx、cy。
/// EstimateDistorted 再估計徑向 k1。不從照片裡找角點。
/// 這不是 calibrate_cameras，也不改 Python 已經算出的九個參數。k2 和切向畸變仍是 0。
/// </summary>
public readonly record struct Intrinsics(double Fx, double Fy, double Cx, double Cy, double K1 = 0)
{
    public double K2 => 0;
    public double P1 => 0;
    public double P2 => 0;
    public double K3 => 0;
}

public static class PlanarCalib
{
    public readonly record struct View(double[] Column, double[] Row);

    public static Intrinsics Estimate(double[] objectX, double[] objectY, IReadOnlyList<View> views)
    {
        if (objectX == null || objectY == null || views == null)
            throw new ArgumentNullException(nameof(objectX));
        if (objectX.Length != objectY.Length || objectX.Length < 4)
            throw new ArgumentException("平面點至少要 4 個");
        if (views.Count < 3)
            throw new ArgumentException("至少要三個視角");
        var equations = new List<double[]>();
        foreach (var view in views)
        {
            if (view.Column.Length != objectX.Length || view.Row.Length != objectX.Length)
                throw new ArgumentException("每個視角的點數要和物點一樣");
            double[,] homography = Homography(objectX, objectY, view.Column, view.Row);
            if (homography[2, 2] < 0)
            {
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        homography[r, c] = -homography[r, c];
            }
            double[] h1 = { homography[0, 0], homography[1, 0], homography[2, 0] };
            double[] h2 = { homography[0, 1], homography[1, 1], homography[2, 1] };
            equations.Add(Vij(h1, h2));
            double[] v11 = Vij(h1, h1);
            double[] v22 = Vij(h2, h2);
            var difference = new double[6];
            for (int i = 0; i < 6; i++)
                difference[i] = v11[i] - v22[i];
            equations.Add(difference);
        }
        var normal = new double[6, 6];
        foreach (double[] row in equations)
        {
            for (int i = 0; i < 6; i++)
                for (int j = 0; j < 6; j++)
                    normal[i, j] += row[i] * row[j];
        }
        double[] b = SmallestEigenvector(normal, 6);
        double b11 = b[0], b12 = b[1], b22 = b[2], b13 = b[3], b23 = b[4], b33 = b[5];
        double denom = b11 * b22 - b12 * b12;
        if (Math.Abs(denom) < 1e-12 || Math.Abs(b11) < 1e-12)
            throw new InvalidOperationException("這些視角分不出焦距和主點");
        double cy = (b12 * b13 - b11 * b23) / denom;
        double lambda = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11;
        if (lambda / b11 <= 0 || lambda * b11 / denom <= 0)
        {
            for (int i = 0; i < 6; i++)
                b[i] = -b[i];
            b11 = b[0]; b12 = b[1]; b22 = b[2]; b13 = b[3]; b23 = b[4]; b33 = b[5];
            denom = b11 * b22 - b12 * b12;
            cy = (b12 * b13 - b11 * b23) / denom;
            lambda = b33 - (b13 * b13 + cy * (b12 * b13 - b11 * b23)) / b11;
        }
        if (lambda / b11 <= 0 || denom == 0 || lambda * b11 / denom <= 0)
            throw new InvalidOperationException("這些視角分不出焦距和主點");
        double fx = Math.Sqrt(lambda / b11);
        double fy = Math.Sqrt(lambda * b11 / denom);
        double skew = -b12 * fx * fx * fy / lambda;
        double cx = skew * cy / fy - b13 * fx * fx / lambda;
        return new Intrinsics(fx, fy, cx, cy);
    }

    /// <summary>
    /// 觀測點帶徑向畸變時，先用直線模型起步，再一起修焦距、主點、k1 和每個視角的姿態。
    /// 模型是 x' = x (1 + k1 r²)。k2 和切向畸變仍是 0。
    /// </summary>
    public static Intrinsics EstimateDistorted(double[] objectX, double[] objectY, IReadOnlyList<View> views)
    {
        Intrinsics seed = Estimate(objectX, objectY, views);
        int poseCount = views.Count * 6;
        var parameters = new double[5 + poseCount];
        parameters[0] = seed.Fx;
        parameters[1] = seed.Fy;
        parameters[2] = seed.Cx;
        parameters[3] = seed.Cy;
        parameters[4] = 0;
        for (int viewIndex = 0; viewIndex < views.Count; viewIndex++)
        {
            double[,] homography = Homography(objectX, objectY, views[viewIndex].Column, views[viewIndex].Row);
            if (homography[2, 2] < 0)
            {
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++)
                        homography[r, c] = -homography[r, c];
            }
            Pose(homography, seed, out double[,] rotation, out double tx, out double ty, out double tz);
            ToRodrigues(rotation, out double rx, out double ry, out double rz);
            int at = 5 + viewIndex * 6;
            parameters[at] = rx;
            parameters[at + 1] = ry;
            parameters[at + 2] = rz;
            parameters[at + 3] = tx;
            parameters[at + 4] = ty;
            parameters[at + 5] = tz;
        }
        double[] residual = Reproject(parameters, objectX, objectY, views);
        double error = Dot(residual, residual);
        double damping = 1e-3;
        for (int iteration = 0; iteration < 12; iteration++)
        {
            double[,] jacobian = Jacobian(parameters, objectX, objectY, views, residual);
            int count = parameters.Length;
            var normal = new double[count, count];
            var right = new double[count];
            for (int i = 0; i < count; i++)
            {
                for (int k = 0; k < residual.Length; k++)
                    right[i] += jacobian[k, i] * residual[k];
                for (int j = 0; j < count; j++)
                {
                    double sum = 0;
                    for (int k = 0; k < residual.Length; k++)
                        sum += jacobian[k, i] * jacobian[k, j];
                    normal[i, j] = sum;
                }
                normal[i, i] += damping * Math.Max(1e-6, Math.Abs(normal[i, i]));
            }
            if (!Solve(normal, right, out double[] step))
                break;
            var trial = new double[count];
            for (int i = 0; i < count; i++)
                trial[i] = parameters[i] - step[i];
            if (trial[0] < 1 || trial[1] < 1)
            {
                damping *= 4;
                continue;
            }
            double[] nextResidual = Reproject(trial, objectX, objectY, views);
            double nextError = Dot(nextResidual, nextResidual);
            if (nextError < error)
            {
                parameters = trial;
                residual = nextResidual;
                error = nextError;
                damping = Math.Max(damping * 0.3, 1e-8);
                if (error < 1e-12)
                    break;
            }
            else
            {
                damping *= 4;
            }
        }
        return new Intrinsics(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4]);
    }

    static double[] Reproject(double[] parameters, double[] objectX, double[] objectY, IReadOnlyList<View> views)
    {
        var residual = new double[objectX.Length * views.Count * 2];
        int at = 0;
        double fx = parameters[0], fy = parameters[1], cx = parameters[2], cy = parameters[3], k1 = parameters[4];
        for (int viewIndex = 0; viewIndex < views.Count; viewIndex++)
        {
            int pose = 5 + viewIndex * 6;
            double[,] rotation = FromRodrigues(parameters[pose], parameters[pose + 1], parameters[pose + 2]);
            double tx = parameters[pose + 3], ty = parameters[pose + 4], tz = parameters[pose + 5];
            var view = views[viewIndex];
            for (int i = 0; i < objectX.Length; i++)
            {
                double x = rotation[0, 0] * objectX[i] + rotation[0, 1] * objectY[i] + tx;
                double y = rotation[1, 0] * objectX[i] + rotation[1, 1] * objectY[i] + ty;
                double z = rotation[2, 0] * objectX[i] + rotation[2, 1] * objectY[i] + tz;
                double idealX = x / z;
                double idealY = y / z;
                double factor = 1 + k1 * (idealX * idealX + idealY * idealY);
                residual[at++] = fx * idealX * factor + cx - view.Column[i];
                residual[at++] = fy * idealY * factor + cy - view.Row[i];
            }
        }
        return residual;
    }

    static double[,] Jacobian(double[] parameters, double[] objectX, double[] objectY, IReadOnlyList<View> views, double[] residual)
    {
        var jacobian = new double[residual.Length, parameters.Length];
        var trial = (double[])parameters.Clone();
        for (int i = 0; i < parameters.Length; i++)
        {
            double step = 1e-6 * Math.Max(1, Math.Abs(parameters[i]));
            trial[i] = parameters[i] + step;
            double[] shifted = Reproject(trial, objectX, objectY, views);
            for (int row = 0; row < residual.Length; row++)
                jacobian[row, i] = (shifted[row] - residual[row]) / step;
            trial[i] = parameters[i];
        }
        return jacobian;
    }

    static double Dot(double[] left, double[] right)
    {
        double sum = 0;
        for (int i = 0; i < left.Length; i++)
            sum += left[i] * right[i];
        return sum;
    }

    static bool Solve(double[,] matrix, double[] right, out double[] answer)
    {
        int size = right.Length;
        var rows = new double[size, size + 1];
        for (int r = 0; r < size; r++)
        {
            for (int c = 0; c < size; c++)
                rows[r, c] = matrix[r, c];
            rows[r, size] = right[r];
        }
        for (int col = 0; col < size; col++)
        {
            int pivot = col;
            double best = Math.Abs(rows[col, col]);
            for (int row = col + 1; row < size; row++)
            {
                double value = Math.Abs(rows[row, col]);
                if (value > best)
                {
                    best = value;
                    pivot = row;
                }
            }
            if (best < 1e-14)
            {
                answer = new double[size];
                return false;
            }
            if (pivot != col)
            {
                for (int c = col; c <= size; c++)
                {
                    double swap = rows[col, c];
                    rows[col, c] = rows[pivot, c];
                    rows[pivot, c] = swap;
                }
            }
            double div = rows[col, col];
            for (int c = col; c <= size; c++)
                rows[col, c] /= div;
            for (int row = 0; row < size; row++)
            {
                if (row == col)
                    continue;
                double factor = rows[row, col];
                if (factor == 0)
                    continue;
                for (int c = col; c <= size; c++)
                    rows[row, c] -= factor * rows[col, c];
            }
        }
        answer = new double[size];
        for (int i = 0; i < size; i++)
            answer[i] = rows[i, size];
        return true;
    }

    static void ToRodrigues(double[,] rotation, out double rx, out double ry, out double rz)
    {
        double trace = rotation[0, 0] + rotation[1, 1] + rotation[2, 2];
        double cosine = Math.Max(-1, Math.Min(1, (trace - 1) / 2));
        double angle = Math.Acos(cosine);
        if (angle < 1e-8)
        {
            rx = ry = rz = 0;
            return;
        }
        double scale = angle / (2 * Math.Sin(angle));
        rx = scale * (rotation[2, 1] - rotation[1, 2]);
        ry = scale * (rotation[0, 2] - rotation[2, 0]);
        rz = scale * (rotation[1, 0] - rotation[0, 1]);
    }

    static double[,] FromRodrigues(double rx, double ry, double rz)
    {
        double angle = Math.Sqrt(rx * rx + ry * ry + rz * rz);
        if (angle < 1e-12)
            return new double[,] { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        double x = rx / angle, y = ry / angle, z = rz / angle;
        double cosine = Math.Cos(angle);
        double sine = Math.Sin(angle);
        double one = 1 - cosine;
        return new double[,]
        {
            { cosine + x * x * one, x * y * one - z * sine, x * z * one + y * sine },
            { y * x * one + z * sine, cosine + y * y * one, y * z * one - x * sine },
            { z * x * one - y * sine, z * y * one + x * sine, cosine + z * z * one },
        };
    }

    static void Pose(double[,] homography, Intrinsics intrinsics, out double[,] rotation, out double tx, out double ty, out double tz)
    {
        var inverse = new double[,]
        {
            { 1 / intrinsics.Fx, 0, -intrinsics.Cx / intrinsics.Fx },
            { 0, 1 / intrinsics.Fy, -intrinsics.Cy / intrinsics.Fy },
            { 0, 0, 1 },
        };
        double[,] axes = Multiply(inverse, homography);
        double Norm(int column)
        {
            double sum = axes[0, column] * axes[0, column] + axes[1, column] * axes[1, column] + axes[2, column] * axes[2, column];
            return Math.Sqrt(sum);
        }
        double scale = 2 / (Norm(0) + Norm(1));
        double r1x = axes[0, 0] * scale, r1y = axes[1, 0] * scale, r1z = axes[2, 0] * scale;
        double r2x = axes[0, 1] * scale, r2y = axes[1, 1] * scale, r2z = axes[2, 1] * scale;
        double r1 = Math.Sqrt(r1x * r1x + r1y * r1y + r1z * r1z);
        r1x /= r1; r1y /= r1; r1z /= r1;
        double dot = r1x * r2x + r1y * r2y + r1z * r2z;
        r2x -= dot * r1x; r2y -= dot * r1y; r2z -= dot * r1z;
        double r2 = Math.Sqrt(r2x * r2x + r2y * r2y + r2z * r2z);
        r2x /= r2; r2y /= r2; r2z /= r2;
        double r3x = r1y * r2z - r1z * r2y;
        double r3y = r1z * r2x - r1x * r2z;
        double r3z = r1x * r2y - r1y * r2x;
        rotation = new[,]
        {
            { r1x, r2x, r3x },
            { r1y, r2y, r3y },
            { r1z, r2z, r3z },
        };
        tx = axes[0, 2] * scale;
        ty = axes[1, 2] * scale;
        tz = axes[2, 2] * scale;
    }

    static double[] Vij(double[] hi, double[] hj)
    {
        return new[]
        {
            hi[0] * hj[0],
            hi[0] * hj[1] + hi[1] * hj[0],
            hi[1] * hj[1],
            hi[2] * hj[0] + hi[0] * hj[2],
            hi[2] * hj[1] + hi[1] * hj[2],
            hi[2] * hj[2],
        };
    }

    static double[,] Homography(double[] objectX, double[] objectY, double[] column, double[] row)
    {
        Normalize(objectX, objectY, out double[] x, out double[] y, out double[,] objectMap);
        Normalize(column, row, out double[] u, out double[] v, out double[,] imageMap);
        int count = x.Length;
        var normal = new double[9, 9];
        for (int i = 0; i < count; i++)
        {
            AddRow(normal, new[] { x[i], y[i], 1, 0, 0, 0, -u[i] * x[i], -u[i] * y[i], -u[i] });
            AddRow(normal, new[] { 0, 0, 0, x[i], y[i], 1, -v[i] * x[i], -v[i] * y[i], -v[i] });
        }
        double[] h = SmallestEigenvector(normal, 9);
        var homography = new double[3, 3];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                homography[r, c] = h[r * 3 + c];
        return Multiply(Multiply(InverseNormalize(imageMap), homography), objectMap);
    }

    static void AddRow(double[,] normal, double[] row)
    {
        for (int i = 0; i < row.Length; i++)
            for (int j = 0; j < row.Length; j++)
                normal[i, j] += row[i] * row[j];
    }

    static void Normalize(double[] x, double[] y, out double[] nx, out double[] ny, out double[,] map)
    {
        int count = x.Length;
        double meanX = 0, meanY = 0;
        for (int i = 0; i < count; i++)
        {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= count;
        meanY /= count;
        double distance = 0;
        for (int i = 0; i < count; i++)
            distance += Math.Sqrt((x[i] - meanX) * (x[i] - meanX) + (y[i] - meanY) * (y[i] - meanY));
        distance /= count;
        double scale = distance < 1e-9 ? 1 : Math.Sqrt(2) / distance;
        nx = new double[count];
        ny = new double[count];
        for (int i = 0; i < count; i++)
        {
            nx[i] = scale * (x[i] - meanX);
            ny[i] = scale * (y[i] - meanY);
        }
        map = new double[,]
        {
            { scale, 0, -scale * meanX },
            { 0, scale, -scale * meanY },
            { 0, 0, 1 },
        };
    }

    static double[,] InverseNormalize(double[,] map)
    {
        double scale = map[0, 0];
        double tx = map[0, 2];
        double ty = map[1, 2];
        return new double[,]
        {
            { 1 / scale, 0, -tx / scale },
            { 0, 1 / scale, -ty / scale },
            { 0, 0, 1 },
        };
    }

    static double[,] Multiply(double[,] left, double[,] right)
    {
        var result = new double[3, 3];
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                result[r, c] = left[r, 0] * right[0, c] + left[r, 1] * right[1, c] + left[r, 2] * right[2, c];
        return result;
    }

    static double[] SmallestEigenvector(double[,] source, int size)
    {
        var matrix = new double[size, size];
        var basis = new double[size, size];
        for (int i = 0; i < size; i++)
        {
            basis[i, i] = 1;
            for (int j = 0; j < size; j++)
                matrix[i, j] = source[i, j];
        }
        for (int iteration = 0; iteration < 80; iteration++)
        {
            int p = 0, q = 1;
            double largest = 0;
            for (int i = 0; i < size; i++)
            {
                for (int j = i + 1; j < size; j++)
                {
                    double value = Math.Abs(matrix[i, j]);
                    if (value > largest)
                    {
                        largest = value;
                        p = i;
                        q = j;
                    }
                }
            }
            if (largest < 1e-14)
                break;
            double app = matrix[p, p];
            double aqq = matrix[q, q];
            double apq = matrix[p, q];
            double phi = 0.5 * Math.Atan2(2 * apq, aqq - app);
            double cosine = Math.Cos(phi);
            double sine = Math.Sin(phi);
            for (int i = 0; i < size; i++)
            {
                if (i == p || i == q)
                    continue;
                double aip = matrix[i, p];
                double aiq = matrix[i, q];
                matrix[i, p] = matrix[p, i] = cosine * aip - sine * aiq;
                matrix[i, q] = matrix[q, i] = sine * aip + cosine * aiq;
            }
            matrix[p, p] = cosine * cosine * app - 2 * sine * cosine * apq + sine * sine * aqq;
            matrix[q, q] = sine * sine * app + 2 * sine * cosine * apq + cosine * cosine * aqq;
            matrix[p, q] = matrix[q, p] = 0;
            for (int i = 0; i < size; i++)
            {
                double vip = basis[i, p];
                double viq = basis[i, q];
                basis[i, p] = cosine * vip - sine * viq;
                basis[i, q] = sine * vip + cosine * viq;
            }
        }
        int chosen = 0;
        for (int i = 1; i < size; i++)
        {
            if (matrix[i, i] < matrix[chosen, chosen])
                chosen = i;
        }
        var vector = new double[size];
        double norm = 0;
        for (int i = 0; i < size; i++)
        {
            vector[i] = basis[i, chosen];
            norm += vector[i] * vector[i];
        }
        norm = Math.Sqrt(norm);
        if (norm > 0)
        {
            for (int i = 0; i < size; i++)
                vector[i] /= norm;
        }
        return vector;
    }
}
