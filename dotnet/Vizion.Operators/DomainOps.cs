namespace Vizion;

public static partial class Operators
{
    public static ImageBuffer Constant(int width, int height)
    {
        if (width < 1 || height < 1)
            throw new ArgumentOutOfRangeException(nameof(width), "寬高至少要 1");
        return ImageBuffer.Wrap(width, height, 1, new byte[width * height]);
    }

    public static ImageBuffer Proto(ImageBuffer image, int gray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int value = ClampByte(gray);
        var pixels = new byte[image.Pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
            pixels[i] = (byte)value;
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, pixels);
    }

    public static ImageBuffer ReduceDomain(ImageBuffer image, Region region) => Restrict(image, region, replace: false);

    public static ImageBuffer ReduceDomain(ImageBuffer image, IReadOnlyList<Region> regions) => Restrict(image, Union(regions), replace: false);

    public static ImageBuffer ChangeDomain(ImageBuffer image, Region region) => Restrict(image, region, replace: true);

    public static ImageBuffer ChangeDomain(ImageBuffer image, IReadOnlyList<Region> regions) => Restrict(image, Union(regions), replace: true);

    public static Region DomainOf(ImageBuffer image)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Domain != null)
            return image.Domain;
        return Rectangle(0, 0, image.Height - 1, image.Width - 1);
    }

    public static ImageBuffer Crop(ImageBuffer image, int row1, int column1, int row2, int column2)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        int top = Math.Max(0, row1);
        int left = Math.Max(0, column1);
        int bottom = Math.Min(image.Height - 1, row2);
        int right = Math.Min(image.Width - 1, column2);
        if (bottom < top || right < left)
            throw new ArgumentException("裁切範圍是空的");
        int height = bottom - top + 1;
        int width = right - left + 1;
        var pixels = new byte[height * width * image.Channels];
        int rowBytes = width * image.Channels;
        for (int row = 0; row < height; row++)
        {
            int source = ((top + row) * image.Width + left) * image.Channels;
            Buffer.BlockCopy(image.Pixels, source, pixels, row * rowBytes, rowBytes);
        }
        Region? domain = image.Domain == null ? null : CropRuns(image.Domain, top, left, bottom, right, width, height);
        return ImageBuffer.Wrap(width, height, image.Channels, pixels, domain);
    }

    public static ImageBuffer SetGray(ImageBuffer image, int row, int column, int gray)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (row < 0 || column < 0 || row >= image.Height || column >= image.Width)
            throw new ArgumentOutOfRangeException(nameof(row), "座標超出影像");
        var pixels = CopyPixels(image);
        int value = ClampByte(gray);
        if (image.Channels == 1)
            pixels[row * image.Width + column] = (byte)value;
        else
        {
            int offset = (row * image.Width + column) * 3;
            pixels[offset] = pixels[offset + 1] = pixels[offset + 2] = (byte)value;
        }
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, pixels, image.Domain);
    }

    public static ImageBuffer Paint(Region region, ImageBuffer image, int gray, string mode) =>
        Paint(new[] { region }, image, gray, mode, respectDomain: false);

    public static ImageBuffer Paint(IReadOnlyList<Region> regions, ImageBuffer image, int gray, string mode) =>
        Paint(regions, image, gray, mode, respectDomain: false);

    public static ImageBuffer Overpaint(Region region, ImageBuffer image, int gray, string mode) =>
        Paint(new[] { region }, image, gray, mode, respectDomain: true);

    public static ImageBuffer Overpaint(IReadOnlyList<Region> regions, ImageBuffer image, int gray, string mode) =>
        Paint(regions, image, gray, mode, respectDomain: true);

    static ImageBuffer Restrict(ImageBuffer image, Region region, bool replace)
    {
        if (image == null || region == null)
            throw new ArgumentNullException(nameof(image));
        if (image.Width != region.Width || image.Height != region.Height)
            throw new ArgumentException("影像和區域的尺寸不同");
        Region domain = replace || image.Domain == null ? region : Intersection(image.Domain, region);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, CopyPixels(image), domain);
    }

    static ImageBuffer Paint(IReadOnlyList<Region> regions, ImageBuffer image, int gray, string mode, bool respectDomain)
    {
        if (regions == null || regions.Count == 0 || image == null)
            throw new ArgumentNullException(nameof(image));
        if (regions[0].Width != image.Width || regions[0].Height != image.Height)
            throw new ArgumentException("區域和影像的尺寸要相同");
        string kind = (mode ?? "").Trim().Trim('\'').ToLowerInvariant();
        if (kind != "fill" && kind != "margin")
            throw new ArgumentException("Type 只能是 fill 或 margin");
        Region area = regions.Count == 1 ? regions[0] : Union(regions);
        if (respectDomain && image.Domain != null)
            area = Intersection(area, image.Domain);
        if (kind == "margin")
            area = Outline(area);
        var pixels = CopyPixels(image);
        int value = ClampByte(gray);
        for (int i = 0; i < area.Rows.Length; i++)
        {
            int row = area.Rows[i];
            int start = area.Columns[i];
            int length = area.Lengths[i];
            if (image.Channels == 1)
            {
                int offset = row * image.Width + start;
                for (int column = 0; column < length; column++)
                    pixels[offset + column] = (byte)value;
            }
            else
            {
                int offset = (row * image.Width + start) * 3;
                for (int column = 0; column < length; column++)
                {
                    pixels[offset + column * 3] = (byte)value;
                    pixels[offset + column * 3 + 1] = (byte)value;
                    pixels[offset + column * 3 + 2] = (byte)value;
                }
            }
        }
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, pixels, image.Domain);
    }

    static Region Outline(Region region)
    {
        if (region.Rows.Length == 0)
            return region;
        var mask = new byte[region.Width * region.Height];
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int offset = region.Rows[i] * region.Width + region.Columns[i];
            for (int column = 0; column < region.Lengths[i]; column++)
                mask[offset + column] = 1;
        }
        var runs = new List<(int Row, int Start, int End)>();
        int width = region.Width;
        int height = region.Height;
        for (int row = 0; row < height; row++)
        {
            int column = 0;
            while (column < width)
            {
                if (!Edge(mask, width, height, row, column))
                {
                    column++;
                    continue;
                }
                int start = column;
                column++;
                while (column < width && Edge(mask, width, height, row, column))
                    column++;
                runs.Add((row, start, column - 1));
            }
        }
        var rows = new int[runs.Count];
        var columns = new int[runs.Count];
        var lengths = new int[runs.Count];
        for (int i = 0; i < runs.Count; i++)
        {
            rows[i] = runs[i].Row;
            columns[i] = runs[i].Start;
            lengths[i] = runs[i].End - runs[i].Start + 1;
        }
        return new Region(width, height, rows, columns, lengths);
    }

    static bool Edge(byte[] mask, int width, int height, int row, int column)
    {
        int index = row * width + column;
        if (mask[index] == 0)
            return false;
        if (row == 0 || mask[index - width] == 0)
            return true;
        if (row == height - 1 || mask[index + width] == 0)
            return true;
        if (column == 0 || mask[index - 1] == 0)
            return true;
        if (column == width - 1 || mask[index + 1] == 0)
            return true;
        return false;
    }

    static Region CropRuns(Region domain, int top, int left, int bottom, int right, int width, int height)
    {
        var rows = new List<int>();
        var columns = new List<int>();
        var lengths = new List<int>();
        for (int i = 0; i < domain.Rows.Length; i++)
        {
            int row = domain.Rows[i];
            if (row < top || row > bottom)
                continue;
            int start = Math.Max(domain.Columns[i], left);
            int end = Math.Min(domain.Columns[i] + domain.Lengths[i] - 1, right);
            if (start > end)
                continue;
            rows.Add(row - top);
            columns.Add(start - left);
            lengths.Add(end - start + 1);
        }
        return new Region(width, height, rows.ToArray(), columns.ToArray(), lengths.ToArray());
    }

    static byte[] CopyPixels(ImageBuffer image)
    {
        var copy = new byte[image.Pixels.Length];
        Buffer.BlockCopy(image.Pixels, 0, copy, 0, copy.Length);
        return copy;
    }

    static int ClampByte(int value)
    {
        if (value < 0)
            return 0;
        if (value > 255)
            return 255;
        return value;
    }
}
