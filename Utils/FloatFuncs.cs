namespace Utils
{
    public class FloatFuncs
    {
        public static float[] Betas(int steps, float k)
        {
            var fret = new float[steps + 1];
            double km = k;
            for(float b = -steps/2; b<steps/2 +1; b++)
            {
                var dex = (int)(b + steps / 2);
                fret[dex] = b;
            }

            return fret;
        }


    }
}
