using MathNet.Numerics.LinearAlgebra;

namespace RedesNeuronales.Resources
{
    public class InputUtils
    {
        public string FormatVector(Vector<double> vector)
        {
            string result = "[ ";
            
            for (int i = 0; i < vector.Count; i++)
            {
                result += vector[i] + " ";
            }

            result += "]";

            return result;
        }

        public Vector<double> GetVectorFromUser(int vectorSize)
        {
            List<double> row = Console.ReadLine()!.Split(' ').Select(double.Parse).ToList();
            Vector<double> vector = Vector<double>.Build.Dense(vectorSize);

            for (int j = 0; j < row.Count; j++)
            {
                vector[j] = row[j];
            }

            return vector;
        }
    }
}
