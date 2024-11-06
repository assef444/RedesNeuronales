using MathNet.Numerics.LinearAlgebra;

namespace RedesNeuronales.Resources.Classes
{
    public class Pattern
    {
        public string Name { get; set; }
        public Vector<double> Vector { get; set; }

        public Pattern(string name, Vector<double> vector)
        {
            Name = name;
            Vector = vector;
        }
    }
}
