namespace Vizion;

public static partial class Operators
{
    /// <summary>會產生一張新影像，不改原來那張。一個灰階可以套到所有點。彩色給三個數時依序是紅、綠、藍。</summary>
    /// <remarks>改指定像素的灰階</remarks>
    /// <example>
    /// <code>Vizion.set_grayval (Image, ImageResult, 10, 12, 255)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="row">列。</param>
    /// <param name="column">行。</param>
    /// <param name="gray">灰階。彩色會塗進三個通道。</param>
    /// <returns>改過指定像素的新影像。不改原來那張。</returns>
    public static ImageBuffer SetGrayval(ImageBuffer image, int row, int column, int gray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (row < 0 || column < 0 || row >= image.Height || column >= image.Width)
            throw new ArgumentException("座標超出影像");
        int value = gray < 0 ? 0 : gray > 255 ? 255 : gray;
        var pixels = new byte[image.Pixels.Length];
        Buffer.BlockCopy(image.Pixels, 0, pixels, 0, pixels.Length);
        if (image.Channels == 1)
            pixels[row * image.Width + column] = (byte)value;
        else
        {
            int offset = (row * image.Width + column) * 3;
            pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = (byte)value;
        }
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, pixels, image.Domain);
    }

    /// <summary>byte 是 0 到 255 的整數。real 是浮點數。</summary>
    /// <remarks>讀出影像型別</remarks>
    /// <example>
    /// <code>Vizion.get_image_type (Image, Type)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>這版的影像都是 byte。</returns>
    public static string ImageType(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return "byte";
    }

    /// <summary>三張都要同樣尺寸。順序是紅、綠、藍。</summary>
    /// <remarks>三張灰階合成彩色</remarks>
    /// <example>
    /// <code>Vizion.channels_to_image (ImageRed, ImageGreen, ImageBlue, MultiChannelImage)</code>
    /// </example>
    /// <param name="red">紅。</param>
    /// <param name="green">綠。</param>
    /// <param name="blue">藍。</param>
    /// <returns>彩色影像。和 compose3 相同。</returns>
    public static ImageBuffer ChannelsToImage(ImageBuffer red, ImageBuffer green, ImageBuffer blue) =>
        Compose3(red, green, blue);

    /// <summary>四邊各加上 Expansion 圈，新像素填 Gray。原本的像素位置往右下移。彩色時三個通道都填同一個灰階。</summary>
    /// <remarks>把影像連定義域一起外擴</remarks>
    /// <example>
    /// <code>Vizion.expand_domain_gray (Image, ImageExpanded, 1, 9)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="expansion">往外加幾圈。</param>
    /// <param name="gray">新像素的灰階。</param>
    /// <returns>擴大後的影像。擴 0 圈且已有定義域時，定義域會變成整張。</returns>
    public static ImageBuffer ExpandDomainGray(ImageBuffer image, int expansion, double gray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (expansion < 0)
            throw new ArgumentException("Expansion 不能是負的");
        int fill = RoundByte(gray);
        ImageBuffer framed = AddImageBorder(image, expansion, fill);
        if (image.Domain == null)
            return framed;
        if (expansion == 0)
            return ImageBuffer.Wrap(framed.Width, framed.Height, framed.Channels, framed.Pixels, Rectangle(0, 0, image.Height - 1, image.Width - 1));
        int width = image.Width + expansion * 2;
        int height = image.Height + expansion * 2;
        var shiftedRows = new int[image.Domain.Rows.Length];
        var shiftedColumns = new int[image.Domain.Rows.Length];
        for (int index = 0; index < image.Domain.Rows.Length; index++)
        {
            shiftedRows[index] = image.Domain.Rows[index] + expansion;
            shiftedColumns[index] = image.Domain.Columns[index] + expansion;
        }
        var shifted = new Region(width, height, shiftedRows, shiftedColumns, image.Domain.Lengths);
        return ImageBuffer.Wrap(width, height, image.Channels, framed.Pixels, Union(shifted, ExpansionFrame(width, height, image.Width, image.Height, expansion)));
    }

    /// <summary>矩形含邊界。矩形外的像素保持原樣。彩色先轉成亮度再等化。</summary>
    /// <remarks>只在一個矩形裡做直方圖等化</remarks>
    /// <example>
    /// <code>Vizion.equ_histo_image_rect (Image, ImageEquHisto, 1, 1, 2, 2)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="row1">上，含。</param>
    /// <param name="column1">左，含。</param>
    /// <param name="row2">下，含。</param>
    /// <param name="column2">右，含。</param>
    /// <returns>只在矩形裡等化後的影像。</returns>
    public static ImageBuffer EquHistoImageRect(ImageBuffer image, int row1, int column1, int row2, int column2)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (row2 < row1 || column2 < column1)
            throw new ArgumentException("矩形座標不正確");
        if (row1 < 0 || column1 < 0 || row2 >= image.Height || column2 >= image.Width)
            throw new ArgumentException("矩形超出影像");
        var destination = new byte[image.Pixels.Length];
        Buffer.BlockCopy(image.Pixels, 0, destination, 0, destination.Length);
        if (!NativeCore.TryEqualizeRect(destination, destination, image.Width, image.Height, image.Channels, row1, column1, row2, column2))
            EqualizeRectScalar(destination, image.Width, image.Height, image.Channels, row1, column1, row2, column2);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>結果是所有走到邊界的路裡面，路上最高灰階的最小值。邊界像素就是自己的灰階。八鄰域。彩色先轉成亮度。</summary>
    /// <remarks>每個像素到邊界要跨過的最低灰階</remarks>
    /// <example>
    /// <code>Vizion.gray_inside (Image, ImageInside)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <returns>每個像素走到邊界的最低瓶頸。</returns>
    public static ImageBuffer GrayInside(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        return CompassGray(image, (source, destination) =>
        {
            if (NativeCore.TryGrayInside(source, destination, image.Width, image.Height))
                return;
            GrayInsideScalar(source, destination, image.Width, image.Height);
        });
    }

    /// <summary>每個像素換成圓裡面排第 Rank 的灰階，從 0 開始。半徑 1 的圓有 5 個像素，Rank 4 就是最亮。邊界用邊緣的像素補上。彩色先轉成亮度。</summary>
    /// <remarks>圓形排序濾波</remarks>
    /// <example>
    /// <code>Vizion.rank_image (Image, ImageRank, 1, 0)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="radius">圓的半徑，1 到 5。</param>
    /// <param name="rank">0 是最暗，最後一個是最亮。</param>
    /// <returns>圓形排序濾波結果。不保留定義域。</returns>
    public static ImageBuffer RankImage(ImageBuffer image, int radius, int rank)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (radius < 1 || radius > 5)
            throw new ArgumentException("半徑要是 1 到 5");
        int count = DiskCount(radius);
        if (rank < 0 || rank >= count)
            throw new ArgumentException("Rank 要在 0 到 " + (count - 1));
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        if (!NativeCore.TryRankDisk(gray, destination, image.Width, image.Height, radius, rank))
            RankDiskScalar(gray, destination, image.Width, image.Height, radius, rank);
        return ImageBuffer.Gray(image.Width, image.Height, destination);
    }

    /// <summary>灰階是 Alpha×(列-Row) + Beta×(行-Column) + Gamma。byte 再四捨五入並夾到 0 到 255。real 保留超過這個範圍的值。</summary>
    /// <remarks>做出一張傾斜的灰階平面</remarks>
    /// <example>
    /// <code>Vizion.gen_image_surface_first_order (ImageSurface, 'byte', 2, 3, 10, 0, 0, 4, 2)</code>
    /// </example>
    /// <param name="kind">只做 byte。</param>
    /// <param name="alpha">列的斜率。</param>
    /// <param name="beta">行的斜率。</param>
    /// <param name="gamma">常數項。</param>
    /// <param name="row">列的原點。</param>
    /// <param name="column">行的原點。</param>
    /// <param name="width">寬度。</param>
    /// <param name="height">高度。</param>
    /// <returns>一次曲面。沒有定義域。</returns>
    public static ImageBuffer SurfaceFirst(string kind, double alpha, double beta, double gamma, double row, double column, int width, int height)
    {
        if (!string.Equals(kind, "byte", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("這版只做 byte");
        if (width < 1 || height < 1)
            throw new ArgumentException("寬高至少要 1");
        var destination = new byte[width * height];
        if (!NativeCore.TrySurface(destination, width, height, alpha, beta, gamma, row, column))
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                    destination[y * width + x] = RoundD(alpha * (y - row) + beta * (x - column) + gamma);
            }
        }
        return ImageBuffer.Gray(width, height, destination);
    }

    /// <summary>列、行先減掉原點。灰階是列平方、行平方、列乘行、列、行和常數這六項相加。byte 再四捨五入並夾到 0 到 255。real 保留超過這個範圍的值。</summary>
    /// <remarks>做出一張彎曲的灰階曲面</remarks>
    /// <example>
    /// <code>Vizion.gen_image_surface_second_order (ImageSurface, 'byte', 1, 0, 0, 0, 0, 4, 0, 0, 4, 3)</code>
    /// </example>
    /// <param name="kind">只做 byte。</param>
    /// <param name="alpha">列的平方項。</param>
    /// <param name="beta">行的平方項。</param>
    /// <param name="gamma">列乘行。</param>
    /// <param name="delta">列的一次項。</param>
    /// <param name="epsilon">行的一次項。</param>
    /// <param name="zeta">常數項。</param>
    /// <param name="row">列的原點。</param>
    /// <param name="column">行的原點。</param>
    /// <param name="width">寬度。</param>
    /// <param name="height">高度。</param>
    /// <returns>二次曲面。沒有定義域。</returns>
    public static ImageBuffer SurfaceSecond(
        string kind, double alpha, double beta, double gamma, double delta, double epsilon, double zeta,
        double row, double column, int width, int height)
    {
        if (!string.Equals(kind, "byte", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("這版只做 byte");
        if (width < 1 || height < 1)
            throw new ArgumentException("寬高至少要 1");
        var destination = new byte[width * height];
        if (!NativeCore.TrySurfaceQuad(destination, width, height, alpha, beta, gamma, delta, epsilon, zeta, row, column))
        {
            for (int y = 0; y < height; y++)
            {
                double yy = y - row;
                for (int x = 0; x < width; x++)
                {
                    double xx = x - column;
                    destination[y * width + x] = RoundD(alpha * yy * yy + beta * xx * xx + gamma * yy * xx + delta * yy + epsilon * xx + zeta);
                }
            }
        }
        return ImageBuffer.Gray(width, height, destination);
    }

    /// <summary>每個物件用自己的平均灰階填滿，物件外是 0。物件重疊時，後面的蓋掉前面的。彩色先轉成亮度，結果是灰階。尺寸要相同。</summary>
    /// <remarks>用區域平均灰階填色</remarks>
    /// <example>
    /// <code>Vizion.region_to_mean (Image, Regions, ImageMean)</code>
    /// </example>
    /// <param name="image">灰階來源。</param>
    /// <param name="regions">要填平均的物件。後面的蓋掉前面的。</param>
    /// <returns>每個物件填上自己的平均。物件外是 0。不保留定義域。</returns>
    public static ImageBuffer RegionToMean(ImageBuffer image, IReadOnlyList<Region> regions)
    {
        if (image == null || regions == null)
            throw new ArgumentNullException(nameof(image));
        byte[] gray = image.GrayBytes();
        var destination = new byte[gray.Length];
        foreach (Region region in regions)
        {
            if (region.Width != image.Width || region.Height != image.Height)
                throw new ArgumentException("影像和區域的尺寸不同");
            if (region.Rows.Length == 0)
                continue;
            if (!NativeCore.TryRegionMean(gray, destination, image.Width, region.Rows, region.Columns, region.Lengths))
                RegionMeanScalar(gray, destination, image.Width, region);
        }
        return ImageBuffer.Gray(image.Width, image.Height, destination);
    }

    static Region ExpansionFrame(int width, int height, int innerWidth, int innerHeight, int ring)
    {
        var rows = new List<int>();
        var columns = new List<int>();
        var lengths = new List<int>();
        for (int row = 0; row < ring; row++)
        {
            rows.Add(row);
            columns.Add(0);
            lengths.Add(width);
            rows.Add(height - ring + row);
            columns.Add(0);
            lengths.Add(width);
        }
        for (int row = 0; row < innerHeight; row++)
        {
            rows.Add(ring + row);
            columns.Add(0);
            lengths.Add(ring);
            rows.Add(ring + row);
            columns.Add(innerWidth + ring);
            lengths.Add(ring);
        }
        return new Region(width, height, rows.ToArray(), columns.ToArray(), lengths.ToArray());
    }

    static int DiskCount(int radius)
    {
        int count = 0;
        for (int dy = -radius; dy <= radius; dy++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                if (dy * dy + dx * dx <= radius * radius)
                    count++;
            }
        }
        return count;
    }

    static void RankDiskScalar(byte[] source, byte[] destination, int width, int height, int radius, int rank)
    {
        var samples = new byte[DiskCount(radius)];
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                int filled = 0;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    int yy = dy < 0 ? (row + dy < 0 ? 0 : row + dy) : (row + dy >= height ? height - 1 : row + dy);
                    for (int dx = -radius; dx <= radius; dx++)
                    {
                        if (dy * dy + dx * dx > radius * radius)
                            continue;
                        int xx = dx < 0 ? (column + dx < 0 ? 0 : column + dx) : (column + dx >= width ? width - 1 : column + dx);
                        byte value = source[yy * width + xx];
                        int index = filled;
                        while (index > 0 && samples[index - 1] > value)
                        {
                            samples[index] = samples[index - 1];
                            index--;
                        }
                        samples[index] = value;
                        filled++;
                    }
                }
                destination[row * width + column] = samples[rank];
            }
        }
    }

    static void RegionMeanScalar(byte[] gray, byte[] destination, int width, Region region)
    {
        double sum = 0;
        int count = 0;
        for (int index = 0; index < region.Rows.Length; index++)
        {
            for (int step = 0; step < region.Lengths[index]; step++)
            {
                sum += gray[region.Rows[index] * width + region.Columns[index] + step];
                count++;
            }
        }
        if (count == 0)
            return;
        byte value = RoundD(sum / count);
        for (int index = 0; index < region.Rows.Length; index++)
        {
            for (int step = 0; step < region.Lengths[index]; step++)
                destination[region.Rows[index] * width + region.Columns[index] + step] = value;
        }
    }

    static void EqualizeRectScalar(byte[] pixels, int width, int height, int channels, int top, int left, int bottom, int right)
    {
        int rectWidth = right - left + 1;
        int rectHeight = bottom - top + 1;
        var crop = new byte[rectWidth * rectHeight];
        if (channels == 1)
        {
            for (int row = 0; row < rectHeight; row++)
                Buffer.BlockCopy(pixels, ((top + row) * width + left), crop, row * rectWidth, rectWidth);
            Equalize(crop, crop, crop.Length);
            for (int row = 0; row < rectHeight; row++)
                Buffer.BlockCopy(crop, row * rectWidth, pixels, ((top + row) * width + left), rectWidth);
            return;
        }
        int count = width * height;
        var y = new byte[count];
        var cr = new byte[count];
        var cb = new byte[count];
        for (int index = 0; index < count; index++)
        {
            int yy, rr, bb;
            BgrToYCrCb(pixels[index * 3], pixels[index * 3 + 1], pixels[index * 3 + 2], out yy, out rr, out bb);
            y[index] = (byte)yy;
            cr[index] = (byte)rr;
            cb[index] = (byte)bb;
        }
        for (int row = 0; row < rectHeight; row++)
            Buffer.BlockCopy(y, (top + row) * width + left, crop, row * rectWidth, rectWidth);
        Equalize(crop, crop, crop.Length);
        for (int row = 0; row < rectHeight; row++)
            Buffer.BlockCopy(crop, row * rectWidth, y, (top + row) * width + left, rectWidth);
        for (int index = 0; index < count; index++)
            YCrCbToBgr(y[index], cr[index], cb[index], pixels, index * 3);
    }

    static void GrayInsideScalar(byte[] source, byte[] destination, int width, int height)
    {
        var result = new int[width * height];
        var seen = new bool[width * height];
        for (int index = 0; index < result.Length; index++)
            result[index] = 1000000000;
        var heap = new SortedSet<(int Value, int Row, int Column)>();
        for (int column = 0; column < width; column++)
        {
            result[column] = source[column];
            result[(height - 1) * width + column] = source[(height - 1) * width + column];
        }
        for (int row = 0; row < height; row++)
        {
            result[row * width] = source[row * width];
            result[row * width + width - 1] = source[row * width + width - 1];
        }
        for (int row = 0; row < height; row++)
        {
            heap.Add((result[row * width], row, 0));
            heap.Add((result[row * width + width - 1], row, width - 1));
        }
        for (int column = 0; column < width; column++)
        {
            heap.Add((result[column], 0, column));
            heap.Add((result[(height - 1) * width + column], height - 1, column));
        }
        int[] drow = { -1, -1, -1, 0, 0, 1, 1, 1 };
        int[] dcol = { -1, 0, 1, -1, 1, -1, 0, 1 };
        while (heap.Count > 0)
        {
            var node = heap.Min;
            heap.Remove(node);
            int index = node.Row * width + node.Column;
            if (seen[index])
                continue;
            seen[index] = true;
            for (int step = 0; step < 8; step++)
            {
                int nrow = node.Row + drow[step];
                int ncol = node.Column + dcol[step];
                if (nrow < 0 || ncol < 0 || nrow >= height || ncol >= width)
                    continue;
                int candidate = Math.Max(node.Value, source[nrow * width + ncol]);
                if (candidate < result[nrow * width + ncol])
                {
                    result[nrow * width + ncol] = candidate;
                    heap.Add((candidate, nrow, ncol));
                }
            }
        }
        for (int index = 0; index < destination.Length; index++)
            destination[index] = (byte)Math.Max(0, Math.Min(255, result[index]));
    }
}
