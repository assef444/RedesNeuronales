using MathNet.Numerics.LinearAlgebra;
using RedesNeuronales.Resources;
using RedesNeuronales.Resources.Classes;
using System.Runtime.Versioning;
using System.Drawing;

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

            List<int> errorCounts = new(); // Lista para almacenar el total de errores por iteración

            int iteration = 0;
            while (!learned)
            {
                #region Training phase
                learned = true;
                int totalErrors = 0; // Contador de errores en la iteración actual

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
                            totalErrors++;
                        }
                    }
                }

                errorCounts.Add(totalErrors); // Guardar el total de errores de esta iteración
                iteration++;
                #endregion
            }

            #region Prints learned weight vector and bias
            Console.Clear();
            Console.WriteLine($"La red neuronal aprendió exitosamente en la iteración {iteration - 1}");
            #endregion

            #region Plot the error graph
            Console.WriteLine("\nGenerando gráfico de errores...");

            // Dimensiones del gráfico
            int width = 800;
            int height = 400;
            Bitmap bmp = new Bitmap(width, height);

            using (Graphics g = Graphics.FromImage(bmp))
            {
                g.Clear(Color.White);

                // Dibujar los ejes
                g.DrawLine(Pens.Black, 50, 350, 750, 350); // Eje X
                g.DrawLine(Pens.Black, 50, 350, 50, 50);   // Eje Y

                // Etiquetas
                g.DrawString("Iteraciones", new Font("Arial", 10), Brushes.Black, 375, 370);
                g.DrawString("Errores", new Font("Arial", 10), Brushes.Black, 5, 200);

                // Escala de datos
                int maxError = errorCounts.Max();
                int pointSpacing = (700 / errorCounts.Count);
                int yScale = (maxError > 0) ? 300 / maxError : 1;

                // Dibujar marcas (ticks) y etiquetas en los ejes
                int numXTicks = 10; // Divisiones en el eje X
                int numYTicks = 10; // Divisiones en el eje Y

                // Ticks y etiquetas del eje X (Iteraciones)
                for (int i = 0; i <= numXTicks; i++)
                {
                    int x = 50 + (i * 700 / numXTicks);
                    int iterLabel = (i * errorCounts.Count / numXTicks);
                    g.DrawLine(Pens.Gray, x, 350, x, 355); // Ticks
                    g.DrawString(iterLabel.ToString(), new Font("Arial", 8), Brushes.Black, x - 10, 360); // Etiquetas
                }

                // Ticks y etiquetas del eje Y (Errores)
                for (int i = 0; i <= numYTicks; i++)
                {
                    int y = 350 - (i * 300 / numYTicks);
                    int errorLabel = (i * maxError / numYTicks);
                    g.DrawLine(Pens.Gray, 45, y, 50, y); // Ticks
                    g.DrawString(errorLabel.ToString(), new Font("Arial", 8), Brushes.Black, 25, y - 5); // Etiquetas
                }

                // Dibujar puntos y líneas
                for (int i = 1; i < errorCounts.Count; i++)
                {
                    int x1 = 50 + (i - 1) * pointSpacing;
                    int y1 = 350 - (errorCounts[i - 1] * yScale);
                    int x2 = 50 + i * pointSpacing;
                    int y2 = 350 - (errorCounts[i] * yScale);

                    g.DrawLine(Pens.Red, x1, y1, x2, y2);
                    g.FillEllipse(Brushes.Blue, x1 - 2, y1 - 2, 4, 4); // Puntos
                }
            }

            string outputPath = Path.Combine(Directory.GetCurrentDirectory(), "error_graph.png");
            bmp.Save(outputPath);
            Console.WriteLine($"Gráfico de errores guardado en: {outputPath}");
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