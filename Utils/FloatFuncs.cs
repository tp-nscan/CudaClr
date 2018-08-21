using System;

namespace Utils
{
    public class FloatFuncs
    {
        public static float[] Betas(int steps, float k)
        {
            var fret = new float[steps + 1];
            double km = k/steps;


           /// float t2 = (float)(1.0 / (1.0 + System.Math.Exp(2 * temp)));

            for (float b = -steps/2; b<steps/2 +1; b++)
            {
                var dex = (int)(b + steps / 2);
                fret[dex] = (float)(1.0 / (1.0 + Math.Exp(km * b)));
            }

            return fret;
        }


    }
}
