using System;
using System.Linq;
using System.Reactive.Subjects;
using Sponge.Model;
using FS;
using Sponge.Common;
using System.Collections.Generic;
using System.Windows.Media;
using Utils;

namespace Sponge.ViewModel.Common
{
    public class UpdateGGRVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
                = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateGGRVm(uint width, uint height, float beta, float flipEnergy, float qRate, 
                                Func<object, ProcResult> proc, Action<object> update_params)
        {
            LatticeDisplayBounds = new R<uint>(
                    minX: 0,
                    maxX: (width > 1000) ? 1000 : width,
                    minY: 0,
                    maxY: (height > 1000) ? 1000 : height
                );

            UpdateVm = new UpdateVm(proc: proc, containingVm: this, update_params: update_params)
            {
                StepsPerUpdate = 100
            };

            GraphLatticeTempVm = new GraphLatticeVm(LatticeDisplayBounds, "Temps", "", "");

            GraphLatticeFlipVm = new GraphLatticeVm(LatticeDisplayBounds, "Flips", "", "");
            
            Beta = beta;
            FlipEnergy = flipEnergy;
            Qrate = qRate;

        }

        public R<float> GraphDisplayBounds { get; set; }

        public R<uint> LatticeDisplayBounds { get; set; }

        public UpdateVm UpdateVm { get; private set; }

        public GraphLatticeVm GraphLatticeTempVm { get; private set; }

        public GraphLatticeVm GraphLatticeFlipVm { get; private set; }

        private float _beta;
        public float Beta
        {
            get { return _beta; }
            set
            {
                SetProperty(ref _beta, value);
            }
        }

        private float _flipEnergy;
        public float FlipEnergy
        {
            get { return _flipEnergy; }
            set
            {
                SetProperty(ref _flipEnergy, value);
            }
        }

        private float _totalHeat;
        public float TotalHeat
        {
            get { return _totalHeat; }
            set
            {
                SetProperty(ref _totalHeat, value);
            }
        }

        private float _qRate;
        public float Qrate
        {
            get { return _qRate; }
            set
            {
                SetProperty(ref _qRate, value);
            }
        }
    }



    public static class GGR_ThermoIsing_bp
    {
        public static UpdateGGRVm Thermo()
        {
            var initTemps = SimGridFloatSamples.HiLow(GridSpan, 0.99f, 0.01f);
            var initFlips = SimGridIntSamples.SquareRandBits(GridSpan, 5213);
            var ggRet = new UpdateGGRVm(
                width: GridSpan, 
                height: GridSpan, 
                beta: Beta,
                flipEnergy: FlipEnergy,
                qRate: qRate, 
                proc: ProcH, 
                update_params: UpdateParams);

            ggRet.GraphLatticeFlipVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_int_BW_mod256, initFlips);
            ggRet.GraphLatticeTempVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_float_BW_mod256, initTemps);

            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            ThermalIsing_bp.Init(initTemps.Data, initFlips.Data, initTemps.Width, BlockSize);
            return ggRet;
        }

        public static uint GridSpan = 512;
        public static uint BlockSize = 4;
        public static float qRate = 0.50f;
        public static float Beta = 9.0f;
        public static float FlipEnergy = 0.0f;

        public static ProcResult ProcH(object vm)
        {
            var ggvm = (UpdateGGRVm)vm;
            return ThermalIsing_bp.UpdateH(
                steps: ggvm.UpdateVm.StepsPerUpdate, 
                qRate: ggvm.Qrate, 
                filpEnergy: ggvm.FlipEnergy, 
                beta: ggvm.Beta);
        }

        static void UpdateParams(object o) { }


        public static void UpdateGGView(ProcResult result, UpdateGGRVm ugvm)
        {
            ugvm.GraphLatticeTempVm.Update(result.Data["ThermGrid"]);
            ugvm.GraphLatticeFlipVm.Update(result.Data["FlipGrid"]);
            var th = (float)result.Data["TotalHeat"];
            ugvm.TotalHeat = th;
        }

    }

}
