using Hybridizer.Runtime.CUDAImports;

namespace Sponge.Common
{
    public class HybStuff
    {
        public static int[] CopyIntResidentArray(IntResidentArray ira)
        {
            ira.RefreshHost();
            var outputs = new int[ira.Count];
            for (var i = 0; i < ira.Count; i++)
            {
                outputs[i] = ira[i];
            }
            return outputs;
        }
    }
}
