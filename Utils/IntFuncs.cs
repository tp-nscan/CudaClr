namespace Utils
{
    public class IntFuncs
    {
        public static int ModDiff(int from, int to, int modulus)
        {
            int diff;
            if (to > from)
            {
                diff = to - from;
                return (diff > modulus/2) ? diff - modulus : diff;
            }
            diff = from - to;
            return (diff > modulus / 2) ? modulus - diff : - diff;
        }

        public static int ModTent(int from, int to, int modulus)
        {
            int md = ModDiff(from, to, modulus); 
            int ms = modulus/4;

            if (md > ms) return modulus/2 - md;
            if (md > -ms) return md;
            return - md - modulus/2;
        }

        public static long Factorial(int f)
        {
            long lRet = 1;
            for (var i = 2; i < f + 1; i++)
            {
                lRet *= i;
            }
            return lRet;
        }


        public static long TwoCycleCount(int f)
        {
            long lRet = 1;
            for (var i = 2; i < f + 1; i++)
            {
                lRet *= i;
            }

            for (var i = 2; i < f + 1; i+=2)
            {
                lRet /= 2;
            }

            return lRet;
        }

    }
}
