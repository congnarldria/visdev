using System.IO.Compression;
using System.Text;

namespace Vizion;

/// <summary>讀非交錯的 8 或 16 位元 PNG。彩色轉成 BGR，16 位元除以 257，和 VisDev 的 read_image 相同。</summary>
static class PngImage
{
    static readonly byte[] Signature = { 137, 80, 78, 71, 13, 10, 26, 10 };

    public static ImageBuffer Decode(byte[] file, string name)
    {
        if (file == null || file.Length < 8)
            throw new InvalidDataException($"無法讀取影像 {name}");
        for (int i = 0; i < Signature.Length; i++)
        {
            if (file[i] != Signature[i])
                throw new InvalidDataException($"無法讀取影像 {name}");
        }
        int width = 0;
        int height = 0;
        int bitDepth = 0;
        int colorType = 0;
        var idat = new MemoryStream();
        int cursor = 8;
        var seenHeader = false;
        while (cursor + 8 <= file.Length)
        {
            int length = ReadInt(file, cursor);
            cursor += 4;
            if (length < 0 || cursor + 4 + length + 4 > file.Length)
                throw new InvalidDataException($"無法讀取影像 {name}");
            string type = Encoding.ASCII.GetString(file, cursor, 4);
            cursor += 4;
            int data = cursor;
            cursor += length + 4;
            if (type == "IHDR")
            {
                if (length < 13)
                    throw new InvalidDataException($"無法讀取影像 {name}");
                width = ReadInt(file, data);
                height = ReadInt(file, data + 4);
                bitDepth = file[data + 8];
                colorType = file[data + 9];
                int compression = file[data + 10];
                int filter = file[data + 11];
                int interlace = file[data + 12];
                if (compression != 0 || filter != 0)
                    throw new InvalidDataException($"無法讀取影像 {name}");
                if (interlace != 0)
                    throw new InvalidDataException("這個 PNG 是交錯掃描，C# 讀圖還沒有做");
                seenHeader = true;
            }
            else if (type == "IDAT")
            {
                idat.Write(file, data, length);
            }
            else if (type == "IEND")
            {
                break;
            }
            else if (type.Length == 4 && type[0] >= 'A' && type[0] <= 'Z' && type != "PLTE")
            {
                throw new InvalidDataException($"無法讀取影像 {name}");
            }
        }
        if (!seenHeader || width <= 0 || height <= 0)
            throw new InvalidDataException($"無法讀取影像 {name}");
        if (colorType == 3)
            throw new InvalidDataException("這個 PNG 用色盤，C# 讀圖還沒有做");
        int bpp = BytesPerPixel(bitDepth, colorType);
        byte[] inflated = Inflate(idat.ToArray(), name);
        byte[] raw = Unfilter(inflated, width, height, bpp, name);
        return Pack(raw, width, height, bitDepth, colorType);
    }

    static int BytesPerPixel(int bitDepth, int colorType)
    {
        int samples = colorType switch
        {
            0 => 1,
            2 => 3,
            4 => 2,
            6 => 4,
            _ => throw new InvalidDataException("這個 PNG 的色彩型別還沒有做"),
        };
        if (bitDepth != 8 && bitDepth != 16)
            throw new InvalidDataException($"PNG 位元深度 {bitDepth} 還沒有做");
        return samples * (bitDepth / 8);
    }

    static byte[] Inflate(byte[] zlib, string name)
    {
        try
        {
            using var input = new MemoryStream(zlib, writable: false);
#if NETFRAMEWORK
            if (zlib.Length < 6)
                throw new InvalidDataException($"無法讀取影像 {name}");
            input.Position = 2;
            using var decoder = new DeflateStream(input, CompressionMode.Decompress);
#else
            using var decoder = new ZLibStream(input, CompressionMode.Decompress);
#endif
            using var output = new MemoryStream();
            decoder.CopyTo(output);
            return output.ToArray();
        }
        catch (InvalidDataException)
        {
            throw;
        }
        catch (Exception)
        {
            throw new InvalidDataException($"無法讀取影像 {name}");
        }
    }

    static byte[] Unfilter(byte[] data, int width, int height, int bpp, string name)
    {
        int stride = width * bpp;
        int need = height * (stride + 1);
        if (data.Length < need)
            throw new InvalidDataException($"無法讀取影像 {name}");
        var raw = new byte[height * stride];
        var previous = new byte[stride];
        var current = new byte[stride];
        int source = 0;
        for (int row = 0; row < height; row++)
        {
            int filter = data[source++];
            Buffer.BlockCopy(data, source, current, 0, stride);
            source += stride;
            for (int column = 0; column < stride; column++)
            {
                int left = column >= bpp ? current[column - bpp] : 0;
                int up = row > 0 ? previous[column] : 0;
                int upperLeft = row > 0 && column >= bpp ? previous[column - bpp] : 0;
                int value = current[column];
                value += filter switch
                {
                    0 => 0,
                    1 => left,
                    2 => up,
                    3 => (left + up) / 2,
                    4 => Paeth(left, up, upperLeft),
                    _ => throw new InvalidDataException($"無法讀取影像 {name}"),
                };
                current[column] = (byte)value;
            }
            Buffer.BlockCopy(current, 0, raw, row * stride, stride);
            Buffer.BlockCopy(current, 0, previous, 0, stride);
        }
        return raw;
    }

    static int Paeth(int left, int up, int upperLeft)
    {
        int estimate = left + up - upperLeft;
        int distanceLeft = Math.Abs(estimate - left);
        int distanceUp = Math.Abs(estimate - up);
        int distanceUpperLeft = Math.Abs(estimate - upperLeft);
        if (distanceLeft <= distanceUp && distanceLeft <= distanceUpperLeft)
            return left;
        if (distanceUp <= distanceUpperLeft)
            return up;
        return upperLeft;
    }

    static ImageBuffer Pack(byte[] raw, int width, int height, int bitDepth, int colorType)
    {
        int count = width * height;
        bool wide = bitDepth == 16;
        int samples = colorType switch
        {
            0 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        };
        bool color = colorType == 2 || colorType == 6;
        var pixels = new byte[count * (color ? 3 : 1)];
        int source = 0;
        int target = 0;
        for (int i = 0; i < count; i++)
        {
            if (!color)
            {
                byte gray = wide ? From16(raw[source], raw[source + 1]) : raw[source];
                pixels[target++] = gray;
            }
            else
            {
                int step = wide ? 2 : 1;
                byte red = wide ? From16(raw[source], raw[source + 1]) : raw[source];
                byte green = wide ? From16(raw[source + step], raw[source + step + 1]) : raw[source + 1];
                byte blue = wide ? From16(raw[source + 2 * step], raw[source + 2 * step + 1]) : raw[source + 2];
                pixels[target++] = blue;
                pixels[target++] = green;
                pixels[target++] = red;
            }
            source += samples * (wide ? 2 : 1);
        }
        return ImageBuffer.Wrap(width, height, color ? 3 : 1, pixels);
    }

    static byte From16(byte high, byte low) => (byte)(((high << 8) | low) / 257);

    static int ReadInt(byte[] file, int offset) =>
        (file[offset] << 24) | (file[offset + 1] << 16) | (file[offset + 2] << 8) | file[offset + 3];
}
