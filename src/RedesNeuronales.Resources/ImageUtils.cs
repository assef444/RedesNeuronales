using MathNet.Numerics.LinearAlgebra;
using System.Drawing;
using System.Runtime.Versioning;

namespace RedesNeuronales.Resources
{
    [SupportedOSPlatform("windows")]
    public static class ImageUtils
    {
        public static int ToStandardInput(this int input)
        {
            return input == 255 ? 1 : -1;
        }

        public static Vector<double> GetVectorFromImage(string filePath)
        {
            List<int> colors = new();
            Bitmap img = new(filePath);

            for (int _y = 0; _y < img.Height; _y++)
            {
                for (int x = 0; x < img.Width; x++)
                {
                    int pixelColor = img.GetPixel(x, _y).R;
                    colors.Add(pixelColor.ToStandardInput());
                }
            }

            Vector<double> vector = Vector<double>.Build.Dense(colors.Count);

            for (int i = 0; i < colors.Count; i++)
            {
                vector[i] = colors[i];
            }

            return vector;
        }
    }
}