using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utils.Ga.Parts;

namespace Utils.Ga
{
    public class GaConjOrbit : GaSortingData
    {
        public GaConjOrbit(Dictionary<string, object> data) : base(data)
        {
        }
    }

    public static class GaConjOrbitExt
    {
        public static GaSortingData ToGaConjOrbitData(
            this IRando randy, uint order,
            uint sorterCount, uint sortableCount, uint stageCount,
            double sorterWinRate, double sortableWinRate)
        {
            var d = new Dictionary<string, object>();
            d.SetCurrentStep(0);
            d.SetSeed(randy.NextInt());
            d.SetSorterWinRate(sorterWinRate);
            d.SetSortableWinRate(sortableWinRate);

            return new GaSortingData(d);
        }
    }
}
