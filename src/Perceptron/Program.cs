using MathNet.Numerics.LinearAlgebra;
using RedesNeuronales.Resources;
using RedesNeuronales.Resources.Classes;
using System.Runtime.Versioning;

namespace Perceptron
{
    [SupportedOSPlatform("windows")]
    public class Program
    {
        public static void Main(string[] args)
        {
            int m;
            int n;
            double threshold;
            List<Vector<double>> weightVectors = new();
            List<Vector<double>> biases = new();
            bool learned = false;
            InputUtils input = new();
            int y;
            int error;
            List<PairPattern> patterns = new();
            int numClasses;

            #region Image mode
            int counter = 0;
                
            #region User inputs number of classes, threshold and images path
            Console.Write("Número de clases a clasificar: ");
            numClasses = int.Parse(Console.ReadLine()!);

            Console.Write("Factor de aprendizaje (threshold): ");
            threshold = double.Parse(Console.ReadLine()!);

            Console.Write("Ingrese la ruta del directorio con imágenes: ");
            string imagesPath = Console.ReadLine()!;
            #endregion

            string[] files = Directory.GetFiles(imagesPath);
                
            foreach (var file in files)
            {
                Vector<double> inputVector = ImageUtils.GetVectorFromImage(file);
                string classLabel = Path.GetFileName(file).Split('_')[0];

                #region Builds output vector
                Vector<double> outputVector = Vector<double>.Build.Dense(numClasses, -1);
                int classIndex = int.Parse(classLabel[^1].ToString()); // Extrae el índice desde la etiqueta
                outputVector[classIndex] = 1;
                #endregion

                patterns.Add(new PairPattern(new Pattern($"X{counter}", inputVector), new Pattern($"Y{counter}", outputVector)));

                counter++;
            }

            n = patterns[0].Input.Vector.Count;
            #endregion

            #region Initializes random weight vectors and biases for each class
            for (int i = 0; i < numClasses; i++)
            {
                weightVectors.Add(PerceptronUtils.GenerateRandomWeightVector(n));
                biases.Add(PerceptronUtils.GenerateRandomBias());
            }
            #endregion

            int iteration = 0;
            while (!learned)
            {
                #region Training phase
                learned = true;

                Console.WriteLine(string.Format("[{0:00000}] Calculando iteración.", iteration));

                foreach (var pattern in patterns)
                {
                    Vector<double> inputVector = pattern.Input.Vector;
                    Vector<double> outputVector = pattern.Output.Vector;

                    for (int classIndex = 0; classIndex < numClasses; classIndex++)
                    {
                        int expectedOutput = (int)outputVector[classIndex];
                        int actualOutput = PerceptronUtils.CalculateY(inputVector, weightVectors[classIndex], biases[classIndex]);

                        error = PerceptronUtils.CalculateError(actualOutput, expectedOutput);

                        if (error != 0)
                        {
                            weightVectors[classIndex] = PerceptronUtils.CalculateWeightVector(weightVectors[classIndex], threshold, error, inputVector);
                            biases[classIndex] = PerceptronUtils.CalculateBias(biases[classIndex], threshold, error);
                            learned = false;
                        }
                    }
                }

                iteration++;
                #endregion
            }

            #region Prints learned weight vector and bias
            Console.Clear();
            Console.WriteLine($"La red neuronal aprendió exitosamente en la iteración {iteration - 1}");
            #endregion

            #region Test phase
            Console.WriteLine("\n===================== TEST MODE =====================\n");

            while (true)
            {
                Console.Write("Ingrese ruta del directorio con imágenes a probar: ");
                string testDirectory = Console.ReadLine()!;

                // Obtener todos los archivos del directorio
                string[] testFiles = Directory.GetFiles(testDirectory);

                // Crear una tabla para almacenar resultados
                List<(string FileName, int PredictedClass)> results = new();

                foreach (var file in testFiles)
                {
                    // Leer vector de características de la imagen
                    Vector<double> testVector = ImageUtils.GetVectorFromImage(file);

                    // Calcular activaciones para todas las clases
                    List<double> activations = new();
                    for (int i = 0; i < numClasses; i++)
                    {
                        double activation = (testVector * weightVectors[i]) + biases[i][0];
                        activations.Add(activation);
                    }

                    // Seleccionar la clase con la mayor activación
                    int predictedClass = activations.IndexOf(activations.Max());

                    // Agregar resultado a la tabla
                    results.Add((Path.GetFileName(file), predictedClass));
                }

                // Mostrar resultados en formato tabular
                Console.WriteLine("\nResultados:");
                Console.WriteLine("===============================================");
                Console.WriteLine($"{"Nombre del archivo",-30}{"Clase Predicha",-10}");
                Console.WriteLine("===============================================");

                foreach (var result in results)
                {
                    Console.WriteLine($"{result.FileName,-30}{result.PredictedClass,-10}");
                }

                Console.WriteLine("===============================================");
            }

            #endregion

        }
    }
}