using System;
using System.Reactive.Subjects;
using System.Windows.Media;
using FS;
using Sponge.Common;
using Sponge.Model;
using Sponge.Model.Lattice;
using Sponge.ViewModel.Common;
using Utils;

namespace Sponge.ViewModel.Lattice
{
    public class UpdateGRVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
                = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateGRVm(uint width, uint height, I<float> betaBounds, I<float> energyBounds, 
            float betaDelta, Func<object, ProcResult> proc, Action<object> update_params)
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

            UpdateVm = new UpdateVm(proc: proc, containingVm:this, update_params: update_params)
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



    public static class GR_Annealer
    {
        public static UpdateGRVm Annealer()
        {
            var initData = SimGridIntSamples.SquareRandBits(GridSpan, 5213);
            var ggRet = new UpdateGRVm(
                GridSpan, 
                GridSpan, 
                BetaBoundsW, 
                EnergyBoundsW, 
                BetaDelta, 
                ProcIsingIntBitsEnergy,
                update_params: UpdateParams);

            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_BWR, initData);
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            IsingIntBits.Init(initData.Data, initData.Width);

            return ggRet;
        }

        public static uint GridSpan = 1024;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 2.2f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.675f, max: 1.3f); }
        }

        public static float BetaDelta = 0.02f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGRVm)vm;
            ggvm.SetBeta();
            return IsingIntBits.UpdateE(ggvm.UpdateVm.StepsPerUpdate, ggvm.Beta);
        }

        public static void UpdateGGView(ProcResult result, UpdateGRVm ugvm)
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


        static void UpdateParams(object o) { }
    }


    public static class GR_AnnealerRb
    {
        public static UpdateGRVm AnnealerRb()
        {
            var initData = SimGridIntSamples.SquareRandBits(GridSpan, 5213);
            var ggRet = new UpdateGRVm(GridSpan, GridSpan, BetaBoundsW, EnergyBoundsW, BetaDelta, ProcIsingIntBitsEnergy,
                update_params: UpdateParams);

            //ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_BWR, initData);
            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_float_BW_mod256, initData);
            
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            BlockPick.Init(initData.Data, initData.Width, 8);

            return ggRet;
        }

        public static uint GridSpan = 1024;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 2.2f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.675f, max: 1.3f); }
        }

        public static float BetaDelta = 0.02f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGRVm)vm;
            ggvm.SetBeta();
            return BlockPick.ProcIsingRb(ggvm.UpdateVm.StepsPerUpdate, ggvm.Beta);
        }

        public static void UpdateGGView(ProcResult result, UpdateGRVm ugvm)
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


        static void UpdateParams(object o) { }
    }


    public static class GR_Thermo_dg
    {
        public static UpdateGRVm Thermo()
        {
            var initData = SimGridFloatSamples.RandUniform0_1(GridSpan, 1234);
            var ggRet = new UpdateGRVm(GridSpan, GridSpan, BetaBoundsW, EnergyBoundsW, BetaDelta, ProcIsingIntBitsEnergy,
                update_params: UpdateParams);

            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_float_BW_mod256, initData);
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            Themal_dg.Init(initData.Data, initData.Width);

            return ggRet;
        }

        public static uint GridSpan = 1024;
        public static float Rate = 0.001f;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 2.2f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.675f, max: 1.3f); }
        }

        public static float BetaDelta = 0.02f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGRVm)vm;
            return Themal_dg.UpdateH(ggvm.UpdateVm.StepsPerUpdate, Rate);
        }

        public static void UpdateGGView(ProcResult result, UpdateGRVm ugvm)
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


        static void UpdateParams(object o) { }

    }


    public static class GR_Thermo_bp
    {
        public static UpdateGRVm Thermo()
        {
            var initData = SimGridFloatSamples.RandUniform0_1(GridSpan, 1234);
            var ggRet = new UpdateGRVm(GridSpan, GridSpan, BetaBoundsW, EnergyBoundsW, BetaDelta, ProcIsingIntBitsEnergy,
                update_params: UpdateParams);

            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_float_BW_mod256, initData);
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            Thermal_bp.Init(initData.Data, initData.Width, BlockSize);

            return ggRet;
        }

        public static uint GridSpan = 1024;
        public static uint BlockSize = 4;
        public static float Rate = 0.001f;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 2.2f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.675f, max: 1.3f); }
        }

        public static float BetaDelta = 0.02f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGRVm)vm;
            return Thermal_bp.UpdateH(ggvm.UpdateVm.StepsPerUpdate, Rate);
        }

        public static void UpdateGGView(ProcResult result, UpdateGRVm ugvm)
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


        static void UpdateParams(object o) { }

    }


    public static class GR_ThermoIsing_bp
    {
        public static UpdateGRVm Thermo()
        {
            var initTemps = SimGridFloatSamples.HiLow(GridSpan, 0.99f, 0.01f);
            var initFlips = SimGridIntSamples.SquareRandBits(GridSpan, 5213);
            var ggRet = new UpdateGRVm(GridSpan, GridSpan, BetaBoundsW, EnergyBoundsW, BetaDelta, 
                ProcIsingIntBitsEnergy, update_params: UpdateParams);

            ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_int_BW_mod256, initFlips);
           // ggRet.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_float_BW_mod256, initTemps);
            
            ggRet.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGGView(p, ggRet));

            ThermalIsing_bp.Init(initTemps.Data, initFlips.Data, initTemps.Width, BlockSize, 4456);

            return ggRet;
        }

        public static uint GridSpan = 256;
        public static uint BlockSize = 4;
        public static float qRate = 0.001f;

        public static I<float> EnergyBoundsW
        {
            get { return new I<float>(min: 0.0f, max: 2.2f); }
        }

        public static I<float> BetaBoundsW
        {
            get { return new I<float>(min: 0.675f, max: 1.3f); }
        }

        public static float BetaDelta = 0.02f;
        public static float FlipEnergy = 0.02f;

        public static ProcResult ProcIsingIntBitsEnergy(object vm)
        {
            var ggvm = (UpdateGRVm)vm;
            return ThermalIsing_bp.UpdateH(
                ggvm.UpdateVm.StepsPerUpdate, 
                qRate, 
                FlipEnergy, 
                9.0f);
        }

        public static void UpdateGGView(ProcResult result, UpdateGRVm ugvm)
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

        static void UpdateParams(object o) { }
    }


}
