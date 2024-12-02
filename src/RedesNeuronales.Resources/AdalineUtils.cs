using MathNet.Numerics.LinearAlgebra;

namespace RedesNeuronales.Resources
{
    public class AdalineUtils
    {
        public static Vector<double> NormalizeVector(Vector<double> vector)
        {
            double norm = vector.L2Norm();
            
            return norm > 0 ? vector / norm : vector;
        }
        
        public static double CalculateY(Vector<double> vector, Vector<double> weightVector, Vector<double> bias)
        {
            double normalizedFactor = vector.L2Norm();
            vector = normalizedFactor > 0 ? vector / normalizedFactor : vector;
            
            return ((vector * weightVector) + bias)[0];
        }

        public static double CalculateError(double output, double predictedOutput)
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
