using System;
using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Genome;

namespace SpongeTester
{
    [TestClass]
    public class GenomeStageDimerFixture
    {
        [TestMethod]
        public void TestToGenomeStageDimer()
        {
            const int order = 12;
            var randy = Rando.Standard(1444);
            var gsd = randy.ToGenomeStageDimer(order: order);

            var res1 = gsd.Stage1.Multiply(gsd.Stage1);
            var res2 = gsd.Stage2.Multiply(gsd.Stage2);
            var mod = gsd.Modifier.Multiply(gsd.Modifier);
        }

        [TestMethod]
        public void TestMutate()
        {
            const int order = 12;
            var randy = Rando.Standard(1444);
            var gsd = randy.ToGenomeStageDimer(order: order);

            var gsdm = gsd.Mutate(randy);
            var res1 = gsdm.Stage1.Multiply(gsdm.Stage1);
            var res2 = gsdm.Stage2.Multiply(gsdm.Stage2);
            var mod = gsdm.Modifier.Multiply(gsdm.Modifier);
        }

        [TestMethod]
        public void TestToPhenotype()
        {
            const int order = 12;
            var randy = Rando.Standard(1444);
            var gsd = randy.ToGenomeStageDimer(order: order);

            var pl = gsd.ToPhenotype().ToArray();

            var pl1s = pl[0].Multiply(pl[0]);
            var pl2s = pl[1].Multiply(pl[1]);
        }
    }
}
