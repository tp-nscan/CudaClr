namespace Utils
{
    public static class GridFuncs
    {
        public static int k_Energy4(int[] grid, uint stride)
        {
            int tot = 0;
            for(var i=0; i<stride; i++)
            {
                for (var j = 0; j < stride; j++)
                {
                    var jl = (j - 1 + stride) % stride;
                    var jr = (j + 1 + stride) % stride;
                    var iu = (i - 1 + stride) % stride;
                    var id = (i + 1 + stride) % stride;

                    var c = grid[i * stride + j];

                    tot += c * (grid[iu * stride + j] +
                               grid[id * stride + j] +
                               grid[i * stride + jr] +
                               grid[i * stride + jl]);
                }
            }
            return tot;
        }
    }
}
