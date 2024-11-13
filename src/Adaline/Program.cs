using MathNet.Numerics.LinearAlgebra;
using RedesNeuronales.Resources;
using RedesNeuronales.Resources.Classes;
using System.Runtime.Versioning;

namespace Adaline
{
    [SupportedOSPlatform("windows")]
    public class Program
    {
        public static void Main(string[] args)
        {
            int m;
            int n;
            double threshold;
            double epsilon;
            Vector<double> weightVector;
            Vector<double> bias;
            bool learned = false;
            InputUtils input = new();
            double y;
            double error;
            List<PairPattern> patterns = new();
            List<double> errors = new();

            #region User chooses program mode
            Console.Write("Modo programa: \n1. Modo manual.\n2. Modo imágenes.\nSeleccione una opción: ");
            int programMode = int.Parse(Console.ReadLine()!);
            #endregion

            if (programMode == 1)
            {
                #region Manual mode
                #region User inputs vectors, epsilon and threshold
                Console.Write("Cantidad de vectores a ingresar (M): ");
                m = int.Parse(Console.ReadLine()!);

                Console.Write("Cantidad de neuronas de cada vector de entrada (N): ");
                n = int.Parse(Console.ReadLine()!);

                Console.Write("Factor de aprendizaje (threshold): ");
                threshold = double.Parse(Console.ReadLine()!);

                Console.Write("Error medio (epsilon): ");
                epsilon = double.Parse(Console.ReadLine()!);
                #endregion

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
            }
            else
            {
                #region Image mode
                int counter = 0;
                
                #region User inputs threshold, epsilon and images path
                Console.Write("Factor de aprendizaje (threshold): ");
                threshold = double.Parse(Console.ReadLine()!);

                Console.Write("Error medio (epsilon): ");
                epsilon = double.Parse(Console.ReadLine()!);

                Console.Write("Ingrese la ruta del directorio con imágenes: ");
                string imagesPath = Console.ReadLine()!;
                #endregion

                string[] files = Directory.GetFiles(imagesPath);
                
                foreach (var file in files)
                {
                    Vector<double> inputVector = ImageUtils.GetVectorFromImage(file);
                    
                    #region Builds output vector
                    Vector<double> outputVector = Vector<double>.Build.Dense(1);
                    outputVector[0] = Path.GetFileName(file)[0].ToString() == "A" ? 1 : -1;
                    #endregion

                    patterns.Add(new PairPattern(new Pattern($"X{counter}", inputVector), new Pattern($"Y{counter}", outputVector)));

                    counter++;
                }

                n = patterns[0].Input.Vector.Count;
                #endregion
            }
            
            weightVector = PerceptronUtils.GenerateRandomWeightVector(n);
            bias = PerceptronUtils.GenerateRandomBias();

            #region Prints values
            Console.Clear();
            Console.WriteLine("Vector de pesos aleatorio");
            Console.WriteLine(weightVector.ToVectorString());
            
            Console.WriteLine("Bias aleatorio");
            Console.WriteLine(bias.ToVectorString());

            Console.WriteLine("=======================================================");
            #endregion

            int iteration = 0;
            while (!learned)
            {
                #region Training phase
                errors.Clear();

                Console.WriteLine(string.Format("[{0:00000}] Calculando iteración.", iteration));

                foreach (var pattern in patterns)
                {
                    Vector<double> inputVector = pattern.Input.Vector;
                    int predictedOutput = (int)pattern.Output.Vector[0];
                    
                    y = AdalineUtils.CalculateY(inputVector, weightVector, bias);
                    error = AdalineUtils.CalculateError(y, predictedOutput);
                    weightVector = AdalineUtils.CalculateWeightVector(weightVector, threshold, error, inputVector);
                    bias = AdalineUtils.CalculateBias(bias, threshold, error);

                    errors.Add(error);
                }

                #region Calculates and checks MSE
                double mse = errors.Select(x => x * x).Average();

                if (mse < epsilon)
                {
                    learned = true;
                    break;
                }
                #endregion

                iteration++;
                #endregion
            }

            #region Prints learned weight vector and bias
            Console.Clear();
            Console.WriteLine(string.Format("La red neuronal aprendió exitosamente en la iteración {0}", iteration - 1));
            
            Console.WriteLine("Con un vector de pesos ideal: ");
            Console.WriteLine(weightVector.ToVectorString());

            Console.WriteLine("Y con un bias de peso ideal: ");
            Console.WriteLine(bias.ToVectorString());
            #endregion

            #region Test phase
            Console.WriteLine("\n===================== TEST MODE =====================\n");
            
            while (true)
            {
                Console.WriteLine("=================================================");
                Console.Write("Ingrese ruta de imagen a probar: ");
                string testPath = Console.ReadLine()!;
                Vector<double> testVector = ImageUtils.GetVectorFromImage(testPath);

                y = AdalineUtils.CalculateY(testVector, weightVector, bias);
                string output = y >= 0 ? "A" : "O";
                
                Console.WriteLine($"La red neuronal clasificó la entrada como: {output} ({y}).");
            }
            #endregion
        }
    }
}