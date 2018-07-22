using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Threading;
using System.Linq;
using Sponge.Common;
using Sponge.M2;
using FS;
using Utils;


namespace Sponge.ViewModel.Common
{
    public class UpdateBinVm : UpdateVm<SimGrid<int>>
    {
        public UpdateBinVm(SimGrid<int> simGrid) : base(simGrid)
        {
            GraphLatticeVm = new GraphLatticeVm(
                    new R<uint>(0, simGrid.Width, 0, simGrid.Height),
                    "Title D", "TitleX D", "TitleY D");

            GraphLatticeVm.SetUpdater(DrawGridCell, simGrid);
            //GraphLatticeVm.OnRangeChanged.Subscribe(v => v.Update(Data));
            StepsPerUpdate = 1;
            // IsingBits2a.Init(inputs: simGrid.Data, span: simGrid.Width);
            Beta = 0.5f;
            IsingBits2.Init(inputs: simGrid.Data, span: simGrid.Width);
        }

        protected override ProcResult<SimGrid<int>> Proc(SimGrid<int> state, int steps)
        {
            //return IsingBits2a.Update(steps, temp:1.0f);

            return IsingBits2.Update3(steps, temp: Beta);
        }

        protected override void UpdateUI(ProcResult<SimGrid<int>> result)
        {
            _graphLatticeVm.Update(result.Data);
            Energy = GridFuncs.Energy4(result.Data.Data, result.Data.Width);
            base.UpdateUI(result);
        }

        private GraphLatticeVm _graphLatticeVm;
        public GraphLatticeVm GraphLatticeVm
        {
            get { return _graphLatticeVm; }
            set
            {
                SetProperty(ref _graphLatticeVm, value);
            }
        }


        private object DrawGridCell(P2<int> dataLoc, R<double> imagePatch, object data)
        {
            Color color;
            var sgd = (SimGrid<int>)data;
            var offset = dataLoc.X + dataLoc.Y * sgd.Width;
            if (sgd.Data[offset] < -0.5)
            {
                color = Colors.White;
            }
            else if (sgd.Data[offset] > 0.5)
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

        private int _energy;
        public int Energy
        {
            get { return _energy; }
            protected set
            {
                SetProperty(ref _energy, value);
            }
        }
    }
}
