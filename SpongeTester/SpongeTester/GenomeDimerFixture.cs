using System;
using System.Text;
using System.Collections.Generic;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Utils;
using Utils.Genome;

namespace SpongeTester
{
    /// <summary>
    /// Summary description for GenomeDimerFixture
    /// </summary>
    [TestClass]
    public class GenomeDimerFixture
    {

        [TestMethod]
        public void TestToGenomeDimer()
        {
            var seed = 1223;
            const int order = 5;
            const int stageCount = 20;

            var randy = Rando.Standard(seed);

            var gd = randy.ToGenomeDimer(order: order, stageCount: stageCount);
        }

        [TestMethod]
        public void TestToSorter()
        {
            var seed = 1223;
            const int order = 50;
            const int stageCount = 20;

            var randy = Rando.Standard(seed);
            var gd = randy.ToGenomeDimer(order: order, stageCount: stageCount);
            var sorter = gd.ToSorter();

        }

        [TestMethod]
        public void TestMutate()
        {
            var seed = 1223;
            const int order = 50;
            const int stageCount = 20;

            var randy = Rando.Standard(seed);
            var gd = randy.ToGenomeDimer(order: order, stageCount: stageCount);
            var gdm = gd.Mutate(randy);
            var sorter = gdm.ToSorter();

        }
    }
}
