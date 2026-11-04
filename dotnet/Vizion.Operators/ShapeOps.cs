namespace Vizion;

public static partial class Operators
{
    /// <summary>用二值結構元素對每個像素取最大灰階。參考點是元素外框的中心，影像外重複邊緣。彩色每個通道分開做。</summary>
    /// <remarks>把亮部往外擴。細的暗縫會被旁邊的亮灰階補上。</remarks>
    /// <example>
    /// <code>Vizion.gray_dilation (Image, SE, ImageDilation)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="se">結構元素區域。空的區域不能用。</param>
    /// <returns>膨脹後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayDilation(ImageBuffer image, Region se) => KernelMorph(image, KernelOf(se), 0);

    /// <summary>用二值結構元素對每個像素取最小灰階。參考點是元素外框的中心，影像外重複邊緣。彩色每個通道分開做。</summary>
    /// <remarks>把暗部往外擴。一個 0 會把周圍一起拉低。</remarks>
    /// <example>
    /// <code>Vizion.gray_erosion (Image, SE, ImageErosion)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="se">結構元素區域。空的區域不能用。</param>
    /// <returns>侵蝕後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayErosion(ImageBuffer image, Region se) => KernelMorph(image, KernelOf(se), 1);

    /// <summary>先用結構元素取最小，再取最大。參考點是元素外框的中心，影像外重複邊緣。</summary>
    /// <remarks>去掉比結構元素小的亮點，留下比較粗的亮部。</remarks>
    /// <example>
    /// <code>Vizion.gray_opening (Image, SE, ImageOpening)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="se">結構元素區域。空的區域不能用。</param>
    /// <returns>開運算後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayOpening(ImageBuffer image, Region se) => KernelMorph(image, KernelOf(se), 2);

    /// <summary>先用結構元素取最大，再取最小。參考點是元素外框的中心，影像外重複邊緣。</summary>
    /// <remarks>補上比結構元素小的暗點。中間一個 0 會被周圍的灰階填上。</remarks>
    /// <example>
    /// <code>Vizion.gray_closing (Image, SE, ImageClosing)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="se">結構元素區域。空的區域不能用。</param>
    /// <returns>閉運算後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayClosing(ImageBuffer image, Region se) => KernelMorph(image, KernelOf(se), 3);

    /// <summary>用矩形、菱形或八角形遮罩取最大灰階。八角形和 OpenCV 的橢圓結構元素相同。高寬要是正奇數，影像外重複邊緣。</summary>
    /// <remarks>不先做區域，直接用固定形狀把亮部往外擴。</remarks>
    /// <example>
    /// <code>Vizion.gray_dilation_shape (Image, ImageMax, 3, 3, 'rectangle')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskHeight">遮罩高，正奇數。</param>
    /// <param name="maskWidth">遮罩寬，正奇數。</param>
    /// <param name="maskShape">rectangle、rhombus 或 octagon。</param>
    /// <returns>膨脹後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayDilationShape(ImageBuffer image, int maskHeight, int maskWidth, string maskShape) =>
        KernelMorph(image, ShapeKernel(maskHeight, maskWidth, maskShape), 0);

    /// <summary>用矩形、菱形或八角形遮罩取最小灰階。高寬要是正奇數，影像外重複邊緣。</summary>
    /// <remarks>不先做區域，直接用固定形狀把暗部往外擴。</remarks>
    /// <example>
    /// <code>Vizion.gray_erosion_shape (Image, ImageMin, 3, 3, 'rectangle')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskHeight">遮罩高，正奇數。</param>
    /// <param name="maskWidth">遮罩寬，正奇數。</param>
    /// <param name="maskShape">rectangle、rhombus 或 octagon。</param>
    /// <returns>侵蝕後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayErosionShape(ImageBuffer image, int maskHeight, int maskWidth, string maskShape) =>
        KernelMorph(image, ShapeKernel(maskHeight, maskWidth, maskShape), 1);

    /// <summary>用固定形狀先取最小再取最大。高寬要是正奇數，影像外重複邊緣。</summary>
    /// <remarks>去掉比遮罩小的亮點。</remarks>
    /// <example>
    /// <code>Vizion.gray_opening_shape (Image, ImageOpening, 3, 3, 'rectangle')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskHeight">遮罩高，正奇數。</param>
    /// <param name="maskWidth">遮罩寬，正奇數。</param>
    /// <param name="maskShape">rectangle、rhombus 或 octagon。</param>
    /// <returns>開運算後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayOpeningShape(ImageBuffer image, int maskHeight, int maskWidth, string maskShape) =>
        KernelMorph(image, ShapeKernel(maskHeight, maskWidth, maskShape), 2);

    /// <summary>用固定形狀先取最大再取最小。高寬要是正奇數，影像外重複邊緣。</summary>
    /// <remarks>補上比遮罩小的暗點。</remarks>
    /// <example>
    /// <code>Vizion.gray_closing_shape (Image, ImageClosing, 3, 3, 'rectangle')</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="maskHeight">遮罩高，正奇數。</param>
    /// <param name="maskWidth">遮罩寬，正奇數。</param>
    /// <param name="maskShape">rectangle、rhombus 或 octagon。</param>
    /// <returns>閉運算後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer GrayClosingShape(ImageBuffer image, int maskHeight, int maskWidth, string maskShape) =>
        KernelMorph(image, ShapeKernel(maskHeight, maskWidth, maskShape), 3);

    /// <summary>遮罩裡的前景權重相等，對每個像素做平均。參考點是外框中心，影像外補 0。有 AVX2 時每一列前面用融合乘加，列尾分開乘加，再四捨五入到偶數。彩色每個通道分開做。</summary>
    /// <remarks>用任意形狀平滑，而不是只能用矩形。遮罩外的像素不計入，所以靠近邊緣會變暗。</remarks>
    /// <example>
    /// <code>Vizion.mean_image_shape (Image, Mask, ImageMean)</code>
    /// </example>
    /// <param name="image">輸入影像。</param>
    /// <param name="mask">當遮罩的區域。要有像素。</param>
    /// <returns>平均後的影像。定義域沿用輸入。</returns>
    public static ImageBuffer MeanImageShape(ImageBuffer image, Region mask)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        (byte[] kernel, int height, int width) = KernelOf(mask);
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryMeanKernel(image.Pixels, destination, image.Width, image.Height, image.Channels, kernel, height, width))
            MeanScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, kernel, height, width);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    /// <summary>用索引圖挑一個通道。索引 1 是紅，2 是綠，3 是藍。灰階圖只接受索引 1，其餘像素是 0。</summary>
    /// <remarks>一張索引圖可以在不同位置挑不同通道，不必整張只用同一個通道。</remarks>
    /// <example>
    /// <code>Vizion.select_grayvalues_from_channels (Image, IndexImage, Selected)</code>
    /// </example>
    /// <param name="image">多通道或灰階影像。</param>
    /// <param name="indexImage">通道索引，從 1 開始。彩色索引圖先轉成亮度。</param>
    /// <returns>挑出來的灰階。沒有定義域。</returns>
    public static ImageBuffer SelectGrayvalues(ImageBuffer image, ImageBuffer indexImage)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        if (indexImage == null)
            throw new ArgumentNullException(nameof(indexImage));
        if (image.Width != indexImage.Width || image.Height != indexImage.Height)
            throw new ArgumentException("影像和索引圖的尺寸不同");
        byte[] index = indexImage.GrayBytes();
        var destination = new byte[image.Width * image.Height];
        if (!NativeCore.TrySelectGray(image.Pixels, index, destination, image.Width, image.Height, image.Channels))
            SelectScalar(image.Pixels, index, destination, image.Width, image.Height, image.Channels);
        return ImageBuffer.Wrap(image.Width, image.Height, 1, destination, null);
    }

    static ImageBuffer KernelMorph(ImageBuffer image, (byte[] Kernel, int Height, int Width) kernel, int mode)
    {
        if (image == null)
            throw new ArgumentNullException(nameof(image));
        var destination = new byte[image.Pixels.Length];
        if (!NativeCore.TryGrayKernel(image.Pixels, destination, image.Width, image.Height, image.Channels, kernel.Kernel, kernel.Height, kernel.Width, mode))
            MorphScalar(image.Pixels, destination, image.Width, image.Height, image.Channels, kernel.Kernel, kernel.Height, kernel.Width, mode);
        return ImageBuffer.Wrap(image.Width, image.Height, image.Channels, destination, image.Domain);
    }

    static (byte[] Kernel, int Height, int Width) ShapeKernel(int height, int width, string maskShape)
    {
        if (height < 1 || width < 1 || (height & 1) == 0 || (width & 1) == 0)
            throw new ArgumentException("遮罩的高和寬要是正的奇數");
        int shape = ShapeCode(maskShape);
        var kernel = new byte[height * width];
        if (!NativeCore.TryShapeKernel(kernel, height, width, shape))
            FillShape(kernel, height, width, shape);
        return (kernel, height, width);
    }

    static int ShapeCode(string maskShape)
    {
        if (maskShape == null)
            throw new ArgumentNullException(nameof(maskShape));
        string name = maskShape.Trim().Trim('\'').Trim('"').ToLowerInvariant();
        if (name == "rectangle")
            return 0;
        if (name == "rhombus")
            return 1;
        if (name == "octagon")
            return 2;
        throw new ArgumentException("MaskShape 只能是 rectangle、rhombus 或 octagon");
    }

    static (byte[] Kernel, int Height, int Width) KernelOf(Region region)
    {
        if (region == null)
            throw new ArgumentNullException(nameof(region));
        int top = int.MaxValue;
        int bottom = int.MinValue;
        int left = int.MaxValue;
        int right = int.MinValue;
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int length = region.Lengths[i];
            if (length <= 0)
                continue;
            int row = region.Rows[i];
            int column = region.Columns[i];
            int end = column + length - 1;
            if (row < top)
                top = row;
            if (row > bottom)
                bottom = row;
            if (column < left)
                left = column;
            if (end > right)
                right = end;
        }
        if (top == int.MaxValue)
            throw new ArgumentException("結構元素是空的");
        int height = bottom - top + 1;
        int width = right - left + 1;
        var kernel = new byte[height * width];
        for (int i = 0; i < region.Rows.Length; i++)
        {
            int length = region.Lengths[i];
            if (length <= 0)
                continue;
            int y = region.Rows[i] - top;
            int x = region.Columns[i] - left;
            for (int step = 0; step < length; step++)
                kernel[y * width + x + step] = 1;
        }
        return (kernel, height, width);
    }

    static void FillShape(byte[] kernel, int height, int width, int shape)
    {
        if (shape == 0)
        {
            for (int i = 0; i < kernel.Length; i++)
                kernel[i] = 1;
            return;
        }
        if (shape == 1)
        {
            int centerRow = height / 2;
            int centerColumn = width / 2;
            double denom = centerRow == 0 ? 1.0 : centerRow;
            for (int row = 0; row < height; row++)
            {
                int dist = row - centerRow;
                if (dist < 0)
                    dist = -dist;
                int span = Bankers((1.0 - dist / denom) * centerColumn);
                int begin = centerColumn - span;
                int end = centerColumn + span + 1;
                if (begin < 0)
                    begin = 0;
                if (end > width)
                    end = width;
                for (int column = begin; column < end; column++)
                    kernel[row * width + column] = 1;
            }
            return;
        }
        int radius = height / 2;
        int halfWidth = width / 2;
        double inverse = radius == 0 ? 0.0 : 1.0 / (radius * (double)radius);
        for (int row = 0; row < height; row++)
        {
            int dy = row - radius;
            int ady = dy < 0 ? -dy : dy;
            int begin = 0;
            int end = 0;
            if (ady <= radius)
            {
                double inside = (radius * (double)radius - dy * (double)dy) * inverse;
                if (inside < 0.0)
                    inside = 0.0;
                int reach = (int)(halfWidth * Math.Sqrt(inside) + 0.5);
                begin = halfWidth - reach;
                end = halfWidth + reach + 1;
                if (begin < 0)
                    begin = 0;
                if (end > width)
                    end = width;
            }
            for (int column = begin; column < end; column++)
                kernel[row * width + column] = 1;
        }
    }

    static int Bankers(double value)
    {
        int whole = (int)Math.Floor(value);
        double frac = value - whole;
        if (frac > 0.5)
            return whole + 1;
        if (frac < 0.5)
            return whole;
        return (whole & 1) != 0 ? whole + 1 : whole;
    }

    static void MorphScalar(byte[] source, byte[] destination, int width, int height, int channels, byte[] kernel, int kernelHeight, int kernelWidth, int mode)
    {
        if (mode == 2)
        {
            var middle = new byte[source.Length];
            Extremum(source, middle, width, height, channels, kernel, kernelHeight, kernelWidth, false);
            Extremum(middle, destination, width, height, channels, kernel, kernelHeight, kernelWidth, true);
            return;
        }
        if (mode == 3)
        {
            var middle = new byte[source.Length];
            Extremum(source, middle, width, height, channels, kernel, kernelHeight, kernelWidth, true);
            Extremum(middle, destination, width, height, channels, kernel, kernelHeight, kernelWidth, false);
            return;
        }
        Extremum(source, destination, width, height, channels, kernel, kernelHeight, kernelWidth, mode == 0);
    }

    static void Extremum(byte[] source, byte[] destination, int width, int height, int channels, byte[] kernel, int kernelHeight, int kernelWidth, bool dilate)
    {
        int anchorY = kernelHeight / 2;
        int anchorX = kernelWidth / 2;
        var rows = new List<int>();
        var columns = new List<int>();
        for (int y = 0; y < kernelHeight; y++)
        {
            for (int x = 0; x < kernelWidth; x++)
            {
                if (kernel[y * kernelWidth + x] == 0)
                    continue;
                rows.Add(y - anchorY);
                columns.Add(x - anchorX);
            }
        }
        if (rows.Count == 0)
            throw new ArgumentException("結構元素是空的");
        for (int channel = 0; channel < channels; channel++)
        {
            for (int row = 0; row < height; row++)
            {
                for (int column = 0; column < width; column++)
                {
                    int value = dilate ? 0 : 255;
                    for (int tap = 0; tap < rows.Count; tap++)
                    {
                        int sampleRow = row + rows[tap];
                        int sampleColumn = column + columns[tap];
                        if (sampleRow < 0)
                            sampleRow = 0;
                        else if (sampleRow >= height)
                            sampleRow = height - 1;
                        if (sampleColumn < 0)
                            sampleColumn = 0;
                        else if (sampleColumn >= width)
                            sampleColumn = width - 1;
                        int sample = source[(sampleRow * width + sampleColumn) * channels + channel];
                        if (dilate)
                        {
                            if (sample > value)
                                value = sample;
                        }
                        else if (sample < value)
                        {
                            value = sample;
                        }
                    }
                    destination[(row * width + column) * channels + channel] = (byte)value;
                }
            }
        }
    }

    static void MeanScalar(byte[] source, byte[] destination, int width, int height, int channels, byte[] kernel, int kernelHeight, int kernelWidth)
    {
        int hits = 0;
        for (int i = 0; i < kernel.Length; i++)
        {
            if (kernel[i] != 0)
                hits++;
        }
        if (hits == 0)
            throw new ArgumentException("結構元素是空的");
        float weight = 1f / hits;
        int anchorY = kernelHeight / 2;
        int anchorX = kernelWidth / 2;
        int lanes = (NativeCore.CpuFlags & 2) != 0 || (NativeCore.CpuFlags & 4) != 0 ? 8 : 0;
        int prefix = MeanPrefix(width * channels, lanes);
        for (int row = 0; row < height; row++)
        {
            for (int column = 0; column < width; column++)
            {
                for (int channel = 0; channel < channels; channel++)
                {
                    bool fused = column * channels + channel < prefix;
                    float sum = 0f;
                    for (int y = 0; y < kernelHeight; y++)
                    {
                        int sampleRow = row + y - anchorY;
                        for (int x = 0; x < kernelWidth; x++)
                        {
                            if (kernel[y * kernelWidth + x] == 0)
                                continue;
                            int sampleColumn = column + x - anchorX;
                            float sample = 0f;
                            if ((uint)sampleRow < (uint)height && (uint)sampleColumn < (uint)width)
                                sample = source[(sampleRow * width + sampleColumn) * channels + channel];
                            sum = fused ? Fused(sample, weight, sum) : sum + sample * weight;
                        }
                    }
                    destination[(row * width + column) * channels + channel] = RoundF(sum);
                }
            }
        }
    }

    static int MeanPrefix(int flat, int lanes)
    {
        if (lanes < 1)
            return 0;
        int index = 0;
        int wide = 4 * lanes;
        int mid = 2 * lanes;
        while (index <= flat - wide)
            index += wide;
        if (index <= flat - mid)
            index += mid;
        if (index <= flat - lanes)
            index += lanes;
        return index;
    }

    static float Fused(float sample, float weight, float acc) =>
        (float)((double)sample * weight + acc);

    static void SelectScalar(byte[] source, byte[] index, byte[] destination, int width, int height, int channels)
    {
        int count = width * height;
        if (channels == 1)
        {
            for (int i = 0; i < count; i++)
                destination[i] = index[i] == 1 ? source[i] : (byte)0;
            return;
        }
        for (int i = 0; i < count; i++)
        {
            byte choice = index[i];
            int pixel = i * 3;
            if (choice == 1)
                destination[i] = source[pixel + 2];
            else if (choice == 2)
                destination[i] = source[pixel + 1];
            else if (choice == 3)
                destination[i] = source[pixel];
            else
                destination[i] = 0;
        }
    }
}
