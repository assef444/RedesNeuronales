using MathNet.Numerics.LinearAlgebra;

namespace RedesNeuronales.Resources
{
    public static class HammingUtils
    {
        public static Matrix<double> BuildWeightMatrix(List<Vector<double>> vectors, int m, int n)
        {
            Matrix<double> weightMatrix = Matrix<double>.Build.Dense(m, n); // initializes mxn matrix
            
            for (int row = 0; row < m; row++)
            {
                for (int col = 0; col < n; col++)
                {
                    weightMatrix[row, col] = 0.5 * vectors[row][col];
                }
            }

            return weightMatrix;
        }

        public static Vector<double> BuildBiasMatrix(int m, int n)
        {
            Vector<double> bias = Vector<double>.Build.Dense(m);

            for (int i = 0; i < m; i++)
            {
                bias[i] = (double)n / 2;
            }

            return bias;
        }

        public static double GetEpsilon(int n)
        {
            return (double)1 / (n - 1);
        }

        public static double HammingTransferFunction(double value)
        {
            return value > 1 ? 1 : value < 0 ? 0 : value;
        }

        public static Vector<double> ApplyTransferFunction(this Vector<double> vector)
        {
            for (int i = 0; i < vector.Count; i++)
            {
                vector[i] = HammingTransferFunction(vector[i]);
            }

            return vector;
        }

        public static Vector<double> GetFirstOutput(Vector<double> testVector, int n, Matrix<double> weightMatrix, Vector<double> bias)
        {
            return (1 / (double)n) * ((weightMatrix * testVector) + bias);
        }

        public static Vector<double> HammingFormula(Vector<double> vector, double epsilon)
        {
            int length = vector.Count;
            Vector<double> result = Vector<double>.Build.Dense(length);

            for (int i = 0; i < length; i++)
            {
                double sum = 0;

                for (int j = 0; j < length; j++)
                {
                    if (j != i)
                    {
                        sum += vector[j];
                    }
                }

                result[i] = vector[i] - (epsilon * sum);
            }

            return result;
        }

        public static int? CheckForPositive(Vector<double> vector)
        {
            int? positiveIndex = null;
            int positiveCount = 0;

            for (int i = 0; i < vector.Count; i++)
            {
                if (vector[i] > 0)
                {
                    positiveCount++;
                    if (positiveCount > 1)
                    {
                        return null;
                    }
                    positiveIndex = i;
                }
            }

            return positiveCount == 1 ? positiveIndex : null;
        }

        public static bool HasDuplicateElements(this Vector<double> vector)
        {
            HashSet<double> uniqueElements = new();

            foreach (double value in vector)
            {
                if (!uniqueElements.Add(value))
                {
                    return true;
                }
            }

            return false;
        }
    }
}