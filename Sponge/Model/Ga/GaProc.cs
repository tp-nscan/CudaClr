using System;
using System.Collections.Generic;
using System.Diagnostics;
using Utils;
using Utils.Sorter;
using Utils.Ga;

namespace Sponge.Model.Ga
{
    public static class GaProc
    {
        public static GaSortingData _sortingGaData;

        static IRando randy;

        public static string InitRandomDirectSortingGaData(int seed, uint order, uint sorterCount, 
            uint sortableCount, uint stageCount, double sortableWinRate,
            double sorterWinRate, StageReplacementMode stageReplacementMode)
        {
            randy = Rando.Standard(seed);

            _sortingGaData = randy.ToRandomDirectSortingGaData(
                order: order,
                sorterCount: sorterCount,
                sortableCount: sortableCount,
                stageCount: stageCount,
                sorterWinRate: sorterWinRate,
                sortableWinRate: sortableWinRate,
                stageReplacementMode: stageReplacementMode
            );

            return string.Empty;
        }


        public static ProcResult ProcGa1(int steps)
        {
            var strRet = String.Empty;
            var  _stopwatch = new Stopwatch();
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                _sortingGaData = _sortingGaData.EvolveBothDirect(randy);
            }

            _stopwatch.Stop();

            var dRet = new Dictionary<string, object>();
            dRet.SetGaSortingData(_sortingGaData);
            return new ProcResult(data: dRet,
                                   err: strRet,
                                   stepsCompleted: steps,
                                   time: _stopwatch.ElapsedMilliseconds);
        }


    }
}
