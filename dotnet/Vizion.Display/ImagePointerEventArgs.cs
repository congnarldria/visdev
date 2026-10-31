namespace Vizion;

public sealed class ImagePointerEventArgs : EventArgs
{
    public ImagePointerEventArgs(double row, double column, int gray)
    {
        Row = row;
        Column = column;
        Gray = gray;
    }

    public double Row { get; }
    public double Column { get; }
    public int Gray { get; }
}
