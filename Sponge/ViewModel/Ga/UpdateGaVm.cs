using Sponge.Common;
using Sponge.Model.Ga;
using Sponge.ViewModel.Common;
using System.Reactive.Subjects;
using System;
using System.Collections.ObjectModel;
using Utils;
using Utils.Ga;
using Utils.Sorter;

namespace Sponge.ViewModel.Ga
{
    public class UpdateGaVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
            = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateGaVm(uint width, uint height, uint order,
                            double sorterWinRate, double sortableWinRate, 
                            uint sorterCount, uint stageCount, uint sortableCount,
                            Func<object, ProcResult> proc, Action<object> update_params)
        {
            UpdateVm = new UpdateVm(proc: proc, containingVm: this, update_params: update_params)
            {
                StepsPerUpdate = 1
            };

            UpdateVm.OnUpdateUI.Subscribe(p => KeepUpdating(p));

            Order = order;
            SorterWinRate = sorterWinRate;
            SortableWinRate = sortableWinRate;
            SorterCount = sorterCount;
            SortableCount = sortableCount;
            StageCount = stageCount;
        }

        public UpdateVm UpdateVm { get; private set; }

        private uint _order;
        public uint Order
        {
            get { return _order; }
            set
            {
                SetProperty(ref _order, value);
            }
        }

        private double _sorterWinRate;
        public double SorterWinRate
        {
            get { return _sorterWinRate; }
            set
            {
                SetProperty(ref _sorterWinRate, value);
            }
        }

        private double _sortableWinRate;
        public double SortableWinRate
        {
            get { return _sortableWinRate; }
            set
            {
                SetProperty(ref _sortableWinRate, value);
            }
        }

        private uint _stageCount;
        public uint StageCount
        {
            get { return _stageCount; }
            set
            {
                SetProperty(ref _stageCount, value);
            }
        }

        private uint _sorterCount;
        public uint SorterCount
        {
            get { return _sorterCount; }
            set
            {
                SetProperty(ref _sorterCount, value);
            }
        }

        private uint _sortableCount;
        public uint SortableCount
        {
            get { return _sortableCount; }
            set
            {
                SetProperty(ref _sortableCount, value);
            }
        }


        private ObservableCollection<string> _report = new ObservableCollection<string>();
        public ObservableCollection<string> Report
        {
            get { return _report; }
            set
            {
                SetProperty(ref _report, value);
            }
        }

        void KeepUpdating(ProcResult result)
        {
            _updateUI.OnNext(result);
            var gasd = result.Data.GetGaSortingData();
            Report.Add(gasd.Report());
        }

    }


    public static class UpdateGaVmExt
    {
        //uint width, uint height, float sorterWinRate, float sortableWinRate,
        //uint sorterCount, uint stageCount, uint sortableCount,

        public static uint GridSpan = 1024;
        public static uint SorterCount = 128;
        public static uint SortableCount = 128;
        public static uint StageCount = 20;
        public static uint Order = 256;
        public static double SorterWinRate = 0.25;
        public static double SortableWinRate = 0.25;
        public static float FlipEnergy = 0.0f;
        public static int Seed = 1234;
        public static StageReplacementMode StageReplacementMode = StageReplacementMode.RCTC;


        public static UpdateGaVm Direct()
        {
            var gaRet = new UpdateGaVm(
                order: Order,
                width: GridSpan,
                height: GridSpan,
                sorterWinRate: SorterWinRate,
                sortableWinRate: SortableWinRate,
                sorterCount: SorterCount,
                stageCount: StageCount,
                sortableCount: SortableCount,
                proc: ProcA,
                update_params: UpdateParams);

            GaProc.InitRandomDirectSortingGaData(
                seed: Seed, 
                order: Order, 
                sorterCount: SorterCount,
                sortableCount: SortableCount, 
                stageCount: StageCount, 
                sortableWinRate: SortableWinRate,
                sorterWinRate: SorterWinRate, 
                stageReplacementMode: StageReplacementMode);

            //gaRet.GraphLatticeFlipVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_int_BW_mod256, initFlips);

            return gaRet;
        }


        static void UpdateParams(object o) { }


        public static ProcResult ProcA(object vm)
        {
            var ggvm = (UpdateGaVm)vm;

            return GaProc.ProcGa1(ggvm.UpdateVm.StepsPerUpdate);
        }

        public static void UpdateGGView(ProcResult result, UpdateGaVm ugvm)
        {
            //ugvm.GraphLatticeTempVm.Update(result.Data["ThermGrid"]);
            //ugvm.GraphLatticeFlipVm.Update(result.Data["FlipGrid"]);
            //var th = (float)result.Data["TotalHeat"];
            //ugvm.TotalHeat = th;
        }


    }
}
