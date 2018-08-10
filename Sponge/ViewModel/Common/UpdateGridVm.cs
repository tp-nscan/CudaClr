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
    public class UpdateGridVm
    {

        public UpdateGridVm(SimGrid<int> data)
        {
            UpdateVm = new UpdateVm(proc: Proc)
            {
                StepsPerUpdate = 1
            };

            GraphLatticeVm = new GraphLatticeVm(
                    new R<uint>(0, data.Height, 0, data.Width),
                    "", "", "");

            GraphLatticeVm.SetUpdater(DrawGridCell, data);

            UpdateVm.OnUpdateUI.Subscribe(p => KeepUpdating(p));
            BlockPick.Init(data.Data, data.Width, 16);

        }

        ProcResult Proc(int steps)
        {
            return BlockPick.ProcMarkBlocks(steps);
        }


        public UpdateVm UpdateVm { get; private set; }
        
        public GraphLatticeVm GraphLatticeVm { get; private set; }

        void KeepUpdating(ProcResult result)
        {
            GraphLatticeVm.Update(result.Data["Grid"]);

        }


        private object DrawGridCell(P2<int> dataLoc, R<double> imagePatch, object data)
        {
            var sg = (SimGrid<int>)data;
            Color color;
            var offset = dataLoc.X + dataLoc.Y * sg.Width;
            int res = (sg.Data[offset]) % 256;
            color = Color.FromArgb((byte)res, 0, 0, 0);
            //if (sg.Data[offset] < -0.5)
            //{
            //    color = Colors.White;
            //}
            //else if (sg.Data[offset] > 0.5)
            //{
            //    color = Colors.Black;
            //}
            //else
            //{
            //    color = Colors.Red;
            //}
            return new RV<float, Color>(
                minX: (float)imagePatch.MinX,
                maxX: (float)imagePatch.MaxX,
                minY: (float)imagePatch.MinY,
                maxY: (float)imagePatch.MaxY,
                v: color);
        }

    }
}
