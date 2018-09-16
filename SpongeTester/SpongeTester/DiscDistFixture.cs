using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;

namespace SpongeTester
{
    [TestClass]
    public class DiscDistFixture
    {
        [TestMethod]
        public void TestDiscDist()
        {
            var tcs = new[] { new Tuple<double, int>(2.3, 0),
                new Tuple<double, int>(0.3, 1),
                new Tuple<double, int>(1.3, 2),
                new Tuple<double, int>(1.3, 3),
                new Tuple<double, int>(1.3, 4)

            };

            var dd = new DiscDistrib<int>(tcs);
            var randy = Rando.Standard(1444);

            var res = dd.Draw(randy);
            for (var i = 0; i < 32; i++)
            {
                Console.WriteLine($" {dd.Draw(randy)} ");
            }
        }

    }
}
