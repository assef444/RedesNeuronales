using MathNet.Numerics.LinearAlgebra;

namespace RedesNeuronales.Resources
{
    public class AdalineUtils
    {
        public static double CalculateY(Vector<double> vector, Vector<double> weightVector, Vector<double> bias)
        {
            return ((vector * weightVector) + bias)[0];
        }

        public static double CalculateError(double output, int predictedOutput)
        {
            return predictedOutput - output;
        }

        public static Vector<double> CalculateWeightVector(Vector<double> weightVector, double threshold, double error, Vector<double> errorVector)
        {
            return (weightVector + ((double)threshold * error * errorVector));
        }

        public static Vector<double> CalculateBias(Vector<double> bias, double threshold, double error)
        {
            return CalculateWeightVector(bias, threshold, error, Vector<double>.Build.Dense(1, 1));
        }
    }
}
