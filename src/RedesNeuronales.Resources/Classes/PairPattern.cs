using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RedesNeuronales.Resources.Classes
{
    public class PairPattern
    {
        public Pattern Input { get; set; }
        public Pattern Output { get; set; }

        public PairPattern(Pattern input, Pattern output)
        {
            Input = input;
            Output = output;
        }
    }
}