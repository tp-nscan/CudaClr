using System;
using System.Collections.Generic;
using System.Diagnostics;
using Utils;
using Utils.Sorter;
using Utils.Ga;
using Utils.Ga.Parts;

namespace Sponge.Model.Ga
{
    public static class GaProc
    {
        // GaStageDimerVm, both conj, no recomb
        public static ProcResult Scheme1(int steps, GaData gasd)
        {
            var randy = Rando.Standard(gasd.Data.GetSeed());
            var strRet = String.Empty;
            var  _stopwatch = new Stopwatch();
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                gasd = gasd.EvolveSortablesConjSortersConj(randy);
                gasd.Data.SetCurrentStep(gasd.Data.GetCurrentStep() + 1);
            }

            _stopwatch.Stop();
            gasd.Data.SetSeed(randy.NextInt());
 
            var dRet = new Dictionary<string, object>();
            dRet.SetGaSortingData(gasd);
            return new ProcResult(data: dRet,
                                  err: strRet,
                                  stepsCompleted: steps,
                                  time: _stopwatch.ElapsedMilliseconds);
        }



        // GaStageDimerVm, both conj, recomb
        public static ProcResult Scheme2(int steps, GaData gasd)
        {
            var randy = Rando.Standard(gasd.Data.GetSeed());
            var strRet = String.Empty;
            var _stopwatch = new Stopwatch();
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                gasd = gasd.EvolveSortersConjRecombSortablesConj(randy);
                gasd.Data.SetCurrentStep(gasd.Data.GetCurrentStep() + 1);
            }

            _stopwatch.Stop();
            gasd.Data.SetSeed(randy.NextInt());

            var dRet = new Dictionary<string, object>();
            dRet.SetGaSortingData(gasd);
            return new ProcResult(data: dRet,
                err: strRet,
                stepsCompleted: steps,
                time: _stopwatch.ElapsedMilliseconds);
        }

        public static ProcResult Scheme3(int steps, GaData gasd)
        {
            var randy = Rando.Standard(gasd.Data.GetSeed());
            var strRet = String.Empty;
            var _stopwatch = new Stopwatch();
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                gasd = gasd.EvolveSorterStageDimerConjRecomb_SortableConj(randy);
                gasd.Data.SetCurrentStep(gasd.Data.GetCurrentStep() + 1);
            }

            _stopwatch.Stop();
            gasd.Data.SetSeed(randy.NextInt());

            var dRet = new Dictionary<string, object>();
            dRet.SetGaSortingData(gasd);
            return new ProcResult(data: dRet,
                err: strRet,
                stepsCompleted: steps,
                time: _stopwatch.ElapsedMilliseconds);
        }

        public static ProcResult Scheme4(int steps, GaData gasd)
        {
            var randy = Rando.Standard(gasd.Data.GetSeed());
            var strRet = String.Empty;
            var _stopwatch = new Stopwatch();
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                gasd = gasd.EvolveStageDimerSortersAndSortables(randy);
                gasd.Data.SetCurrentStep(gasd.Data.GetCurrentStep() + 1);
            }

            _stopwatch.Stop();
            gasd.Data.SetSeed(randy.NextInt());

            var dRet = new Dictionary<string, object>();
            dRet.SetGaSortingData(gasd);
            return new ProcResult(data: dRet,
                err: strRet,
                stepsCompleted: steps,
                time: _stopwatch.ElapsedMilliseconds);
        }

        public static ProcResult Scheme5(int steps, GaData gasd)
        {
            var randy = Rando.Standard(gasd.Data.GetSeed());
            var strRet = String.Empty;
            var _stopwatch = new Stopwatch();
            _stopwatch.Reset();
            _stopwatch.Start();

            for (var i = 0; i < steps; i++)
            {
                gasd = gasd.EvolveConjOrbitSortersAndSortables(randy);
                gasd.Data.SetCurrentStep(gasd.Data.GetCurrentStep() + 1);
            }

            _stopwatch.Stop();
            gasd.Data.SetSeed(randy.NextInt());

            var dRet = new Dictionary<string, object>();
            dRet.SetGaSortingData(gasd);
            return new ProcResult(data: dRet,
                err: strRet,
                stepsCompleted: steps,
                time: _stopwatch.ElapsedMilliseconds);
        }

    }
}
