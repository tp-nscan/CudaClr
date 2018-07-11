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
    }
}
