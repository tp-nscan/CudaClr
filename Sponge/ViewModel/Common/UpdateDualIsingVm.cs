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
    public class UpdateDualIsingVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
                = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;


        float _betaMin = 0.0f;
        float _betaMax = 4.0f;

        public UpdateDualIsingVm(SimGrid<int> data)
        {
            UpdateVm = new UpdateVm(proc: Proc, containingVm: this)
            {
                StepsPerUpdate = 1
            };

            Rects = new List<RV<float, Color>>();
            Points = new List<P2V<float, Color>>();

            GraphVm = new GraphVm(new R<float>(0, 3, 0, 4))
            {
                Title = "Energy vs Temp",
                TitleX = "Temp",
                TitleY = "Energy"
            };

            GraphLatticeVm = new GraphLatticeVm(
                                new R<uint>(0, data.Width, 0, data.Height),
                                "", "", "");

            GraphLatticeVm.SetUpdater(DrawGridCell, data);

            Beta = 1.08f;

            UpdateVm.OnUpdateUI.Subscribe(p => KeepUpdating(p));
            IsingIntBits.Init(data.Data, data.Width);
            Beta = _betaMin;
        }

        public List<RV<float, Color>> Rects { get; private set; }
        public List<P2V<float, Color>> Points { get; private set; }

        ProcResult Proc(object steps)
        {
            return IsingIntBits.UpdateE((int)steps, temp: Beta);
        }

        public UpdateVm UpdateVm { get; private set; }

        public GraphVm GraphVm { get; private set; }

        public GraphLatticeVm GraphLatticeVm { get; private set; }

        float _eMin = 2.1f;
        float _eMax = 2.7f;
        float smidgeX;
        float smidgeY;

        void KeepUpdating(ProcResult result)
        {
            smidgeX = (_betaMax - _betaMin) / 500;
            smidgeY = (_eMax - _eMin) / 500;

            var boundingRect = new R<float>(_betaMin, _betaMax, 1.8f, 2.7f);

            Energy = (float)result.Data["Energy"];

            GraphVm.WbImageVm.ImageData = Id.AddRect(
                GraphVm.WbImageVm.ImageData,
                new RV<float, Color>(
                            minX: Beta,
                            maxX: Beta + smidgeX,
                            minY: Energy,
                            maxY: Energy + smidgeY,
                            v: GetColor()
                ));

            GraphLatticeVm.Update(result.Data["Grid"]);
            
            _updateUI.OnNext(result);

        }

        Color GetColor()
        {
            return Colors.Black;
            //if (UpdateVm.StepsPerUpdate < 10)
            //    return Colors.LightBlue;

            //if (UpdateVm.StepsPerUpdate < 100)
            //    return Colors.MediumBlue;

            //if (UpdateVm.StepsPerUpdate < 200)
            //    return Colors.Green;

            //if (UpdateVm.StepsPerUpdate < 400)
            //    return Colors.Red;

            //return Color.FromArgb(255, 0, 0, 0);

        }

        private object DrawGridCell(P2<int> dataLoc, R<double> imagePatch, object data)
        {
            var sg = (SimGrid<int>)data;
            Color color;
            var offset = dataLoc.X + dataLoc.Y * sg.Width;
            if (sg.Data[offset] < -0.5)
            {
                color = Colors.White;
            }
            else if (sg.Data[offset] > 0.5)
            {
                color = Colors.Black;
            }
            else
            {
                color = Colors.Red;
            }
            return new RV<float, Color>(
                minX: (float)imagePatch.MinX,
                maxX: (float)imagePatch.MaxX,
                minY: (float)imagePatch.MinY,
                maxY: (float)imagePatch.MaxY,
                v: color);
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

        private float _energy;
        public float Energy
        {
            get { return _energy; }
            protected set
            {
                SetProperty(ref _energy, value);
            }
        }


    }

}
