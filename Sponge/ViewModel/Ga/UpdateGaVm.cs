using Sponge.Common;
using Sponge.Model.Ga;
using Sponge.ViewModel.Common;
using System.Reactive.Subjects;
using System;
using System.Collections.ObjectModel;
using System.Linq;
using System.Text;
using System.Windows;
using Utils;
using Utils.Ga;
using Utils.Ga.Parts;
using Utils.Sorter;
using System.Windows.Input;

namespace Sponge.ViewModel.Ga
{
    public class UpdateGaVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
            = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateGaVm(GaSortingData gaSortingData,
                          uint width, uint height, uint order,
                          double sorterWinRate, double sortableWinRate, 
                          uint sorterCount, uint stageCount, uint sortableCount,
                          Func<object, ProcResult> proc, 
                          Action<object> update_params)
        {
            UpdateVm = new UpdateVm(proc: proc, containingVm: this, update_params: update_params)
            {
                StepsPerUpdate = 10
            };
            UpdateVm.OnUpdateUI.Subscribe(p => KeepUpdating(p));

            GaSortingData = gaSortingData;
            Order = order;
            SorterWinRate = sorterWinRate;
            SortableWinRate = sortableWinRate;
            SorterCount = sorterCount;
            SortableCount = sortableCount;
            StageCount = stageCount;
        }

        public GaSortingData GaSortingData { get; private set; }

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

        private string _message;
        public string Message
        {
            get { return _message; }
            set
            {
                SetProperty(ref _message, value);
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
            GaSortingData = result.Data.GetGaSortingData();
            var seq = GaSortingData.Data.GetSortablePool().Sortables.Values.First().GetMap();
            Message = StringFuncs.LineFormat(seq);
            Report.Add($"{GaSortingData.Data.GetCurrentStep()}\t{GaSortingData.Report()}");
        }

        #region CopyReportCommand

        private RelayCommand _copyReportCommand;

        public ICommand CopyReportCommand 
            => _copyReportCommand ?? 
               (_copyReportCommand = new RelayCommand(DoCopyReport, CanCopyReport));


        private void DoCopyReport()
        {
            var sb = new StringBuilder();
            foreach (var r in Report)
            {
                sb.AppendLine(r);
            }
            Clipboard.SetText(sb.ToString());
        }

        private bool CanCopyReport()
        {
            return true; // _isRunning;
        }

        #endregion

    }


    public static class UpdateGaVmExt
    {
        //uint width, uint height, float seed, float sortableWinRate,
        //uint sorterCount, uint stageCount, uint sortableCount,

        public static uint GridSpan = 1024;
        public static uint SorterCount = 32;
        public static uint SortableCount = 128;
        public static uint StageCount = 120;
        public static uint Order = 128;
        public static double SorterWinRate = 0.25;
        public static double SortableWinRate = 0.25;
        public static float FlipEnergy = 0.0f;
        public static int Seed = 97266;
        public static StageReplacementMode StageReplacementMode = StageReplacementMode.RCTC;


        public static UpdateGaVm GaStageDimerVm()
        {
            var gasd = GaProc.InitRandomStageDimerGaData(
                seed: Seed,
                order: Order,
                sorterCount: SorterCount,
                sortableCount: SortableCount,
                stageCount: StageCount,
                sortableWinRate: SortableWinRate,
                sorterWinRate: SorterWinRate,
                stageReplacementMode: StageReplacementMode);

            var gaRet = new UpdateGaVm(
                gaSortingData: gasd,
                order: Order,
                width: GridSpan,
                height: GridSpan,
                sorterWinRate: SorterWinRate,
                sortableWinRate: SortableWinRate,
                sorterCount: SorterCount,
                stageCount: StageCount,
                sortableCount: SortableCount,
                proc: ProcScheme3,
                update_params: UpdateParams);

            return gaRet;
        }

        public static UpdateGaVm Direct()
        {
            var gasd = GaProc.InitRandomDirectSortingGaData(
                seed: Seed,
                order: Order,
                sorterCount: SorterCount,
                sortableCount: SortableCount,
                stageCount: StageCount,
                sortableWinRate: SortableWinRate,
                sorterWinRate: SorterWinRate,
                stageReplacementMode: StageReplacementMode);

            var gaRet = new UpdateGaVm(
                gaSortingData: gasd,
                order: Order,
                width: GridSpan,
                height: GridSpan,
                sorterWinRate: SorterWinRate,
                sortableWinRate: SortableWinRate,
                sorterCount: SorterCount,
                stageCount: StageCount,
                sortableCount: SortableCount,
                proc: ProcScheme2,
                update_params: UpdateParams);

            return gaRet;
        }

        static void UpdateParams(object o) { }

        // for Direct
        public static ProcResult ProcScheme1(object vm)
        {
            var ggvm = (UpdateGaVm)vm;
            return GaProc.Scheme1(ggvm.UpdateVm.StepsPerUpdate, ggvm.GaSortingData);
        }

        // for Direct
        public static ProcResult ProcScheme2(object vm)
        {
            var ggvm = (UpdateGaVm)vm;
            return GaProc.Scheme2(ggvm.UpdateVm.StepsPerUpdate, ggvm.GaSortingData);
        }

        // for StageDimer
        public static ProcResult ProcScheme3(object vm)
        {
            var ggvm = (UpdateGaVm)vm;
            return GaProc.Scheme3(ggvm.UpdateVm.StepsPerUpdate, ggvm.GaSortingData);
        }

        // for StageDimer
        public static ProcResult ProcScheme4(object vm)
        {
            var ggvm = (UpdateGaVm)vm;
            return GaProc.Scheme4(ggvm.UpdateVm.StepsPerUpdate, ggvm.GaSortingData);
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
