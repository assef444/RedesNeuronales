using MathNet.Numerics.LinearAlgebra;
using RedesNeuronales.Resources.Classes;
using RedesNeuronales.Resources;

namespace Hamming
{
    public class Program
    {
        public static void Main(string[] args)
        {
            int m;
            int n;
            List<Pattern> patterns = new();
            Matrix<double> weightMatrix;
            Vector<double> bias;
            double epsilon;
            bool converged = false;
            InputUtils input = new();

            #region User inputs vectors
            Console.Write("Cantidad de vectores a ingresar (M): ");
            m = int.Parse(Console.ReadLine()!);

            Console.Write("Cantidad de neuronas de cada vector (N): ");
            n = int.Parse(Console.ReadLine()!);

            for (int i = 0; i < m; i++)
            {
                Console.Write(string.Format("Vector X{0} (separado por espacios): ", i));
                Vector<double> vector = input.GetVectorFromUser(n);

                patterns.Add(new Pattern($"X{i}", vector));
            }

            List<Vector<double>> vectors = patterns.Select(x => x.Vector).ToList(); // extracts Vector property of patterns
            #endregion

            weightMatrix = HammingUtils.BuildWeightMatrix(vectors, m, n);
            bias = HammingUtils.BuildBiasMatrix(m, n);
            epsilon = HammingUtils.GetEpsilon(n);

            Console.Clear();

            #region Print values
            Console.WriteLine("Matriz de pesos");
            Console.WriteLine(weightMatrix.ToMatrixString());
            
            Console.WriteLine("BIAS");
            Console.WriteLine(bias.ToVectorString());
            
            Console.WriteLine("Epsilon");
            Console.WriteLine(epsilon);

            Console.WriteLine("\n=======================================================\n");
            #endregion

            while (true)
            {
                int i = 0;
                
                #region User inputs test vector
                Console.Write("Vector de prueba (separado por espacios): ");
                Vector<double> testVector = input.GetVectorFromUser(n);
                #endregion

                Vector<double> vector = HammingUtils.GetFirstOutput(testVector, n, weightMatrix, bias).ApplyTransferFunction();
                Console.WriteLine($"U({i}) = {input.FormatVector(vector)}");
                
                while (!converged)
                {
                    if (!vector.HasDuplicateElements())
                    {
                        i++;
                    
                        vector = HammingUtils.HammingFormula(vector, epsilon).ApplyTransferFunction();
                        Console.WriteLine($"U({i}) = {input.FormatVector(vector)}");

                        int? positiveIndex = HammingUtils.CheckForPositive(vector);
                    
                        if (positiveIndex != null)
                        {
                            Console.WriteLine($"La RN converge y asocia {input.FormatVector(testVector)} con {patterns[positiveIndex.Value].Name} = {input.FormatVector(patterns[positiveIndex.Value].Vector)}");
                            converged = true;
                        }
                    }
                    else
                    {
                        Console.WriteLine($"La RN no asociará con ningún patrón porque contiene al menos 2 elementos iguales.");
                        converged = true;
                    }
                }

                Console.WriteLine();
                Console.WriteLine("=======================================================");
                converged = false;
            }
        }
    }
}