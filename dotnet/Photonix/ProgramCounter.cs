namespace Photonix;

static class ProgramCounter
{
    public static int LineCount(string text)
    {
        if (string.IsNullOrEmpty(text))
            return 1;
        int count = 1;
        foreach (char ch in text)
        {
            if (ch == '\n')
                count++;
        }
        return count;
    }

    public static int NextStop(int current, int count, IReadOnlySet<int> breakpoints, ref int breakSkip)
    {
        if (count < 1)
        {
            breakSkip = 0;
            return 0;
        }
        current = Math.Clamp(current < 1 ? 1 : current, 1, count);
        if (breakpoints.Contains(current) && breakSkip != current)
        {
            breakSkip = current;
            return current;
        }
        for (int line = current + 1; line <= count; line++)
        {
            if (!breakpoints.Contains(line))
                continue;
            breakSkip = line;
            return line;
        }
        breakSkip = 0;
        return 0;
    }

    public static string FormatTime(double ms, int count)
    {
        string text;
        if (ms < 0.001)
            text = "<0.001 ms";
        else if (ms < 1)
            text = ms.ToString("0.000") + " ms";
        else if (ms < 100)
            text = ms.ToString("0.00") + " ms";
        else if (ms < 1000)
            text = ms.ToString("0.0") + " ms";
        else
            text = (ms / 1000).ToString("0.00") + " s";
        if (count > 1)
            text += " ×" + count;
        return text;
    }
}
