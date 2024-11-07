using MathNet.Numerics.LinearAlgebra;
using RedesNeuronales.Resources.Classes;
using RedesNeuronales.Resources;

namespace Perceptron
{
    public class Program
    {
        public static void Main(string[] args)
        {
            int m;
            int n;
            double threshold;
            Vector<double> weightVector;
            Vector<double> bias;
            bool learned = false;
            InputUtils input = new();
            int y;
            int error;
            List<PairPattern> patterns = new();

            #region User inputs vectors and threshold
            Console.Write("Cantidad de vectores a ingresar (M): ");
            m = int.Parse(Console.ReadLine()!);

            Console.Write("Cantidad de neuronas de cada vector de entrada (N): ");
            n = int.Parse(Console.ReadLine()!);

            Console.Write("Factor de aprendizaje (threshold): ");
            threshold = double.Parse(Console.ReadLine()!);

            for (int i = 0; i < m; i++)
            {
                Console.Write(string.Format("Vector X{0} (separado por espacios): ", i));
                Vector<double> inputVector = input.GetVectorFromUser(n);

                Console.Write(string.Format("Vector Y{0} (separado por espacios): ", i));
                Vector<double> outputVector = input.GetVectorFromUser();

                patterns.Add(new PairPattern(new Pattern($"X{i}", inputVector), new Pattern($"Y{i}", outputVector)));

                Console.WriteLine("========================");
            }
            #endregion
            
            weightVector = PerceptronUtils.GenerateRandomWeightVector(n);
            bias = PerceptronUtils.GenerateRandomBias();
            
            Console.Clear();

            #region Print values
            Console.WriteLine("Vector de pesos aleatorio");
            Console.WriteLine(weightVector.ToVectorString());
            
            Console.WriteLine("Bias aleatorio");
            Console.WriteLine(bias.ToVectorString());

            Console.WriteLine("=======================================================");
            #endregion

            int iteration = 0;
            while (!learned)
            {
                learned = true;

                Console.WriteLine(string.Format("[{0:00000}] Calculando iteración.", iteration));

                foreach (var pattern in patterns)
                {
                    Vector<double> inputVector = pattern.Input.Vector;
                    int predictedOutput = (int)pattern.Output.Vector[0];
                    
                    y = PerceptronUtils.TransferFunction(PerceptronUtils.CalculateY(inputVector, weightVector, bias));
                    error = PerceptronUtils.CalculateError(y, predictedOutput);

                    if (error != 0)
                    {
                        weightVector = PerceptronUtils.CalculateWeightVector(weightVector, threshold, error, inputVector);
                        bias = PerceptronUtils.CalculateBias(bias, threshold, error);
                        learned = false;
                        break;
                    }
                }

                iteration++;
            }

            Console.WriteLine("==============================================================");
            Console.WriteLine(string.Format("La red neuronal aprendió exitosamente en la iteración {0}", iteration - 1));
            Console.WriteLine("Con un vector de pesos ideal: ");
            Console.WriteLine(weightVector.ToVectorString());

            Console.WriteLine("Y con un bias de peso ideal: ");
            Console.WriteLine(bias.ToVectorString());
            Console.ReadKey();
        }
    }
}