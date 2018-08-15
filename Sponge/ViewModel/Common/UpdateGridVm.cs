using System; 
using FS;
using Sponge.Model;
using Utils;

namespace Sponge.ViewModel.Common
{
    public class UpdateGridVm
    {
        public UpdateGridVm(SimGrid<int> data, Func<object, ProcResult> proc)
        {
            UpdateVm = new UpdateVm(proc: proc, containingVm: this)
            {
                StepsPerUpdate = 1
            };

            GraphLatticeVm = new GraphLatticeVm(
                    new R<uint>(0, data.Height, 0, data.Width),
                    "", "", "");
        }

        public UpdateVm UpdateVm { get; private set; }
        
        public GraphLatticeVm GraphLatticeVm { get; private set; }
    }



    public static class UpdateGridVmB
    {
        public static UpdateGridVm BlockPicker()
        {
            var initData = SimGridIntSamples.SquareRandBits(512, 5213);
            var ugvm = new UpdateGridVm(initData, ProcMarkBlocks);

            BlockPick.Init(initData.Data, initData.Width, 4);

            ugvm.GraphLatticeVm.SetUpdater(GraphLatticeVmEx.DrawGridCell_int_BW_mod256, initData);
            ugvm.UpdateVm.OnUpdateUI.Subscribe(p => UpdateGraphLatticeWithGrid(p, ugvm));

            return ugvm;
        }


        public static ProcResult ProcMarkBlocks(object steps)
        {
            return BlockPick.ProcMarkBlocks((int)steps);
        }

        public static void UpdateGraphLatticeWithGrid(ProcResult result, UpdateGridVm ugvm)
        {
            ugvm.GraphLatticeVm.Update(result.Data["Grid"]);
        }

    }
}
