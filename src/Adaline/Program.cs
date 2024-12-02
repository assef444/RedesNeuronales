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
            List<Vector<double>> weightVectors = new();
            List<Vector<double>> biases = new();
            bool learned = false;
            InputUtils input = new();
            double y;
            double error;
            List<PairPattern> patterns = new();
            List<double> errors = new();
            List<string> labels = new()
            {
                "A", "O", "D", "H"
            };
            List<double> classErrors = new(labels.Count);

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
                    Vector<double> outputVector = Vector<double>.Build.Dense(labels.Count);
                    string label = Path.GetFileName(file)[0].ToString();

                    int labelIndex = labels.IndexOf(label);
                    if (labelIndex >= 0)
                    {
                        outputVector[labelIndex] = 1;
                    }
                    else
                    {
                        throw new Exception($"Etiqueta desconocida encontrada en el archivo: {file}");
                    }
                    #endregion

                    patterns.Add(new PairPattern(new Pattern($"X{counter}", inputVector), new Pattern($"Y{counter}", outputVector)));

                    counter++;
                }

                n = patterns[0].Input.Vector.Count;
                #endregion
            }

            #region Initializes weight vector and bias for each label
            for (int i = 0; i < labels.Count; i++)
            {
                weightVectors.Add(PerceptronUtils.GenerateRandomWeightVector(n));
                biases.Add(PerceptronUtils.GenerateRandomBias());
            }
            #endregion

            int iteration = 0;
            while (!learned)
            {
                #region Training phase
                classErrors.Clear();

                Console.WriteLine(string.Format("[{0:00000}] Calculando iteración.", iteration));

                foreach (var pattern in patterns)
                {
                    Vector<double> inputVector = AdalineUtils.NormalizeVector(pattern.Input.Vector);
                    Vector<double> actualOutput = pattern.Output.Vector;

                    for (int classIndex = 0; classIndex < labels.Count; classIndex++)
                    {
                        // Desired output for this class (1 for current class, 0 for others)
                        double desiredOutput = actualOutput[classIndex];

                        // Calculate output for the current Adaline
                        y = AdalineUtils.CalculateY(inputVector, weightVectors[classIndex], biases[classIndex]);

                        // Calculate error for the current class
                        error = AdalineUtils.CalculateError(y, desiredOutput);

                        // Update weights and bias for this class
                        weightVectors[classIndex] = AdalineUtils.CalculateWeightVector(weightVectors[classIndex], threshold, error, inputVector);
                        biases[classIndex] = AdalineUtils.CalculateBias(biases[classIndex], threshold, error);

                        // Add squared error for this class
                        classErrors.Add(error * error);
                    }
                }

                #region Calculates and checks MSE
                double mse = classErrors.Average();
                
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
            #endregion

            #region Test phase
            Console.WriteLine("\n===================== TEST MODE =====================\n");
            
            while (true)
            {
                Console.WriteLine("=================================================");
                Console.Write("Ingrese ruta de imagen a probar: ");
                string testPath = Console.ReadLine()!;
                Vector<double> testVector = ImageUtils.GetVectorFromImage(testPath);
                List<double> outputs = new();

                for (int classIndex = 0; classIndex < labels.Count; classIndex++)
                {
                    y = AdalineUtils.CalculateY(testVector, weightVectors[classIndex], biases[classIndex]);
                    outputs.Add(y); // Guarda la salida
                }
                
                int predictedClassIndex = outputs.IndexOf(outputs.Max());
                string predictedLabel = labels[predictedClassIndex]; // Etiqueta asociada

                Console.WriteLine($"La red neuronal clasificó la entrada como: {predictedLabel} ({outputs[predictedClassIndex]}).");
            }
            #endregion
        }
    }
}