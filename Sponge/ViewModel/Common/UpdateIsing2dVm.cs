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
    public class UpdateIsing2dVm : BindableBase
    {
        private readonly Subject<ProcResult> _updateUI
                = new Subject<ProcResult>();
        public IObservable<ProcResult> OnUpdateUI => _updateUI;

        public UpdateIsing2dVm(SimGrid<int> data)
        {
            UpdateVm = new UpdateVm(proc: Proc)
            {
                StepsPerUpdate = 1
            };

            //Rects = new List<RV<float, Color>>();
            Points = new List<P2V<float, Color>>();

            GraphVm = new GraphVm()
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

            GraphVm.SetData(
                    boundingRect: new R<float>(0, 3, 0, 4),
                    plotPoints: Enumerable.Empty<P2V<float, Color>>(),
                    openRects: Enumerable.Empty<RV<float, Color>>(),
                    filledRects: Enumerable.Empty<RV<float, Color>>(),
                    plotLines: Enumerable.Empty<LS2V<float, Color>>()
                );

            UpdateVm.OnUpdateUI.Subscribe(p => KeepUpdating(p));
            IsingBitsDual.Init(data.Data, data.Width);
            Beta = _betaMin;
        }

        public List<RV<float, Color>> Rects  { get; private set; }
        public List<P2V<float, Color>> Points { get; private set; }

        ProcResult Proc(int steps)
        {
            return IsingBitsDual.UpdateE2(steps, temp: Beta, flip: Flip);
        }

        public UpdateVm UpdateVm { get; private set; }

        public GraphVm GraphVm { get; private set; }

        public GraphLatticeVm GraphLatticeVm { get; private set; }

        float smidge = 0.0002f;

        void KeepUpdating(ProcResult result)
        {
            Energy = (float)result.Data["Energy"];

            if (UpdateVm.TotalSteps < 5000) return;

            //Rects.Add(new RV<float, Color>(
            //                minX: Beta,
            //                maxX: Beta + smidge / 4,
            //                minY: Energy,
            //                maxY: Energy + smidge,
            //                v: GetColor()
            //    ));
            //GraphVm.SetRects(boundingRect: new R<float>(_betaMin, _betaMax, 2.15f, 2.45f), filledRects: Rects);

            Points.Add(new P2V<float, Color>
                (
                    x: Beta,
                    y: Energy,
                    v: GetColor()
                ));

            GraphVm.SetPoints(boundingRect: new R<float>(1.07f, 1.13f, 2.1f, 2.7f), plotPoints: Points);

            GraphLatticeVm.Update(result.Data["Grid"]);

            SetBeta();
            _updateUI.OnNext(result);

        }

        bool _decreasing;
        float _betaMin = 1.07f;
        float _betaMax = 1.13f;
        float _betaDelta = 0.00001f;
        void SetBeta()
        {
            if(_decreasing)
            {
                Beta -= _betaDelta;
                if (Beta < _betaMin)
                    {
                    Beta = _betaMin;
                    _decreasing = false;
                }
            }
            else
            {
                Beta += _betaDelta;
                if (Beta > _betaMax)
                {
                    Beta = _betaMax;
                    _decreasing = true;
                }
            }
        }

        Color GetColor()
        {
            return (_decreasing) ? Colors.Red : Colors.Black;
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

        private bool _flip;
        public bool Flip
        {
            get { return _flip; }
            set
            {
                SetProperty(ref _flip, value);
            }
        }
    }
}
