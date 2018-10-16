using System;
using System.Diagnostics;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Ga;
using Utils.Ga.Parts;
using Utils.Genome;
using Utils.Sortable;
using Utils.Sorter;

namespace SpongeTester
{
    [TestClass]
    public class GaSorterBitsFixture
    {

        public double Mutato1(double mutationRate, double step)
        {
            return mutationRate * (20.0/step);
        }

        public double Mutato2(double mutationRate, double step)
        {
            var mr = mutationRate * (1.0 / Math.Log(step + 10));
            return mr;
        }

        public double Mutato(double step)
        {
            double uni = 4.69219E-05;
            double a = 0.1;
            double b = 0.5;
            double c = 1.0;
            return uni * (a + b / (Math.Log(step + 10) + c));
        }
    }
}
