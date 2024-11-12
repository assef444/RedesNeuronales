using MathNet.Numerics.LinearAlgebra;

namespace RedesNeuronales.Resources
{
    public static class PerceptronUtils
    {
        public static Vector<double> GenerateRandomWeightVector(int size)
        {
            Random random = new();
            double[] values = new double[size];

            for (int i = 0; i < size; i++)
            {
                values[i] = random.NextDouble();
            }

            return Vector<double>.Build.Dense(values);
        }

        public static Vector<double> GenerateRandomBias(int size = 1)
        {
            return GenerateRandomWeightVector(size);
        }

        public static int TransferFunction(double value)
        {
            return value >= 0 ? 1 : -1;
        }
        
        public static int CalculateY(Vector<double> vector, Vector<double> weightVector, Vector<double> bias)
        {
            return TransferFunction(((vector * weightVector) + bias)[0]);
        }

        public static int CalculateError(int output, int predictedOutput)
        {
            return predictedOutput - output;
        }

        public static Vector<double> CalculateWeightVector(Vector<double> weightVector, double threshold, int error, Vector<double> errorVector)
        {
            return (weightVector + ((double)threshold * error * errorVector));
        }

        public static Vector<double> CalculateBias(Vector<double> bias, double threshold, int error)
        {
            return CalculateWeightVector(bias, threshold, error, Vector<double>.Build.Dense(1, 1));
        }
    }
}