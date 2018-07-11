using System.Linq;
using System.Windows.Media;
using Sponge.ViewModel.Common;
using Utils;

namespace Sponge.ViewModel.Design.Common
{
    public class LegendVmD : LegendVm
    {
        public LegendVmD() : base("minVal", "midVal", "maxVal", Colors.Pink,
            TestColors2(), Colors.DeepPink)
        {
        }

        public static Color[] TestColors()
        {
            return new[] {Colors.Red, Colors.Green, Colors.Blue, Colors.Red, Colors.Green, Colors.Blue};
        }

        public static Color[] TestColors2()
        {
            return ColorSeq.ColorTent(1024, 0.0, 0.333, 0.666, 0.4, 0.1, 0.1, true).ToArray();
        }
    }
}
