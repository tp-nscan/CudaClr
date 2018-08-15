using System;
using System.Linq;
using Sponge.Model;
using FS;
using Sponge.Common;
using System.Collections.Generic;
using System.Windows.Media;
using Utils;

namespace Sponge.ViewModel.Common
{
    public class UpdateIsingBpVm : BindableBase
    {
        public UpdateIsingBpVm(SimGrid<int> data)
        {
            UpdateVm = new UpdateVm(proc: Proc, containingVm: this)
            {
                StepsPerUpdate = 1
            };

            Rects = new List<RV<float, Color>>();

            GraphVm = new GraphVm(new R<float>(0, 3, 0, 4))
            {
                Title = "Energy vs Temp",
                TitleX = "Temp",
                TitleY = "Energy"
            };

            GraphLatticeVm = new GraphLatticeVm(
                                new R<uint>(0, 512, 0, 512),
                                "", "", "");

            GraphLatticeVm.SetUpdater(DrawGridCell, data);

            Beta = 1.08f;

            UpdateVm.OnUpdateUI.Subscribe(p => KeepUpdating(p));

            BlockPick.Init(data.Data, data.Width, 4);
            Beta = _betaMin;
        }

        public GraphLatticeVm GraphLatticeVm { get; private set; }

        float _eMin = 1.5f;
        float _eMax = 2.7f;
        float smidgeX;
        float smidgeY;

        float _betaMin = 0.77f;
        float _betaMax = 1.33f;

        void KeepUpdating(ProcResult result)
        {
            smidgeX = (_betaMax - _betaMin) / 500;
            smidgeY = (_eMax - _eMin) / 500;

            var boundingRect = new R<float>(_betaMin, _betaMax, 1.8f, 2.7f);

            Energy = (float)result.Data["Energy"];
            GraphLatticeVm.Update(result.Data["Grid"]);

            GraphVm.WbImageVm.ImageData = Id.AddRect(
              GraphVm.WbImageVm.ImageData,
              new RV<float, Color>(
                          minX: Beta,
                          maxX: Beta + smidgeX,
                          minY: Energy,
                          maxY: Energy + smidgeY,
                          v: GetColor()
              ));

        }


        ProcResult Proc(object steps)
        {
            return BlockPick.ProcIsingRb((int)steps, temp: Beta);
        }


        public List<RV<float, Color>> Rects { get; private set; }

        public UpdateVm UpdateVm { get; private set; }

        public GraphVm GraphVm { get; private set; }


        Color GetColor()
        {
            return Colors.Black;
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


        private object DrawGridCell(P2<int> dataLoc, R<double> imagePatch, object data)
        {
            var sg = (SimGrid<int>)data;
            Color color;
            var offset = dataLoc.X + dataLoc.Y * sg.Width;

            color = (sg.Data[offset] > 0) ? Colors.Black : Colors.White;

            return new RV<float, Color>(
                minX: (float)imagePatch.MinX,
                maxX: (float)imagePatch.MaxX,
                minY: (float)imagePatch.MinY,
                maxY: (float)imagePatch.MaxY,
                v: color);
        }


    }


}
