using System;
using System.Linq;
using System.Reactive.Subjects;
using Sponge.Model;
using FS;
using Sponge.Common;
using System.Collections.Generic;
using System.Windows.Media;

namespace Sponge.ViewModel.Common
{
    public class UpdateGGVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
                = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateGGVm(uint width, uint height, I<float> betaBounds, I<float> energyBounds, float betaDelta, Func<object, ProcResult> proc)
        {
            BetaBounds = betaBounds;
            EnergyBounds = energyBounds;

            LatticeDisplayBounds = new R<uint>(
                    minX: 0, 
                    maxX: (width > 1000) ? 1000 : width,
                    minY: 0, 
                    maxY: (height > 1000) ? 1000 : height
                );

            GraphDisplayBounds = new R<float>(
                    minX: BetaBounds.Min,
                    maxX: BetaBounds.Max,
                    minY: EnergyBounds.Min,
                    maxY: EnergyBounds.Max
                ); 

            UpdateVm = new UpdateVm(proc: proc, containingVm:this)
            {
                StepsPerUpdate = 100
            };

            GraphVm = new GraphVm(GraphDisplayBounds)
            {
                Title = "Energy vs Temp",
                TitleX = "Temp",
                TitleY = "Energy"
            };
            
            GraphLatticeVm = new GraphLatticeVm(LatticeDisplayBounds, "", "", "");
            
            BetaDelta = betaDelta;
            Beta = BetaBounds.Min;
        }

        public I<float> BetaBounds { get; set; }

        public I<float> EnergyBounds { get; set; }

        public R<float> GraphDisplayBounds { get; set; }

        public R<uint> LatticeDisplayBounds { get; set; }

        public List<RV<float, Color>> Rects  { get; private set; }

        public List<P2V<float, Color>> Points { get; private set; }

        public UpdateVm UpdateVm { get; private set; }

        public GraphVm GraphVm { get; private set; }

        public GraphLatticeVm GraphLatticeVm { get; private set; }

        bool _decreasing;
        public bool Decreasing
        {
            get { return _decreasing; }
            set
            {
                SetProperty(ref _decreasing, value);
            }
        }

        public void SetBeta()
        {
            if (BetaDelta == 0) return;

            if (Decreasing)
            {
                Beta -= BetaDelta;
                if (Beta < BetaBounds.Min)
                    {
                    Beta = BetaBounds.Min;
                    Decreasing = false;
                }
            }
            else
            {
                Beta += BetaDelta;
                if (Beta > BetaBounds.Max)
                {
                    Beta = BetaBounds.Max;
                    Decreasing = true;
                }
            }
        }

        private float _beta;
        public float Beta
        {
            get { return _beta; }
            set
            {
                SetProperty(ref _beta, value);
            }
        }

        private float _betaDelta;
        public float BetaDelta
        {
            get { return _betaDelta; }
            set
            {
                SetProperty(ref _betaDelta, value);
            }
        }

        private float _energy;
        public float Energy
        {
            get { return _energy; }
            set
            {
                SetProperty(ref _energy, value);
            }
        }
    }



    public static class GG_Annealer
    {
        public static UpdateGGVm Annealer()
        {
            var initData = SimGridIntSamples.SquareRandBits(GridSpan, 5213);
            var ggRet = new UpdateGGVm(GridSpan, GridSpan, BetaBoundsW, EnergyBoundsW, BetaDelta, ProcIsingIntBitsEnergy);

            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_BWR, initData);
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            IsingIntBits.Init(initData.Data, initData.Width);

            return ggRet;
        }

        public static uint GridSpan = 1024;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 4.0f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.25f, max: 1.5f); }
        }

        public static float BetaDelta = 0.05f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGGVm)vm;
            ggvm.SetBeta();
            return IsingIntBits.UpdateE(ggvm.UpdateVm.StepsPerUpdate, ggvm.Beta);
        }

        public static void UpdateGGView(ProcResult result, UpdateGGVm ugvm)
        {
            ugvm.GraphLatticeVm.Update(result.Data["Grid"]);
            ugvm.Energy = 4.0f - (float)result.Data["Energy"];

            var smidgeX = (ugvm.BetaBounds.Max - ugvm.BetaBounds.Min) / 500;
            var smidgeY = (ugvm.EnergyBounds.Max - ugvm.EnergyBounds.Min) / 500;

            ugvm.GraphVm.WbImageVm.ImageData = Id.AddRect(
                ugvm.GraphVm.WbImageVm.ImageData,
                new RV<float, Color>(
                            minX: ugvm.Beta,
                            maxX: ugvm.Beta + smidgeX,
                            minY: ugvm.Energy,
                            maxY: ugvm.Energy + smidgeY,
                            v: (ugvm.Decreasing) ? Colors.Red : Colors.Black
                ));
        }

    }


    public static class GG_Thermo
    {
        public static UpdateGGVm Thermo()
        {
            var initData = SimGridFloatSamples.LeftRightGradient(GridSpan);
            var ggRet = new UpdateGGVm(GridSpan, GridSpan, BetaBoundsW, EnergyBoundsW, BetaDelta, ProcIsingIntBitsEnergy);

            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_float_BW_mod256, initData);
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            ThemalIsing.Init(initData.Data, initData.Width);

            return ggRet;
        }

        public static uint GridSpan = 64;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 4.0f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.25f, max: 1.5f); }
        }

        public static float BetaDelta = 0.05f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGGVm)vm;
            return ThemalIsing.UpdateH(ggvm.UpdateVm.StepsPerUpdate);
        }

        public static void UpdateGGView(ProcResult result, UpdateGGVm ugvm)
        {
            ugvm.GraphLatticeVm.Update(result.Data["Grid"]);

            //ugvm.Energy = 4.0f - (float)result.Data["Energy"];

            //var smidgeX = (ugvm.BetaBounds.Max - ugvm.BetaBounds.Min) / 500;
            //var smidgeY = (ugvm.EnergyBounds.Max - ugvm.EnergyBounds.Min) / 500;

            //ugvm.GraphVm.WbImageVm.ImageData = Id.AddRect(
            //    ugvm.GraphVm.WbImageVm.ImageData,
            //    new RV<float, Color>(
            //                minX: ugvm.Beta,
            //                maxX: ugvm.Beta + smidgeX,
            //                minY: ugvm.Energy,
            //                maxY: ugvm.Energy + smidgeY,
            //                v: (ugvm.Decreasing) ? Colors.Red : Colors.Black
            //    ));
        }

    }


}
