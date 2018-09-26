using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public static class Bitly
    {
        static Bitly()
        {
            OneHotBytes = new byte[8];
            byte byteBit = 1;
            for (var i = 0; i < 8; i++)
            {
                OneHotBytes[i] = (byte)(byteBit << i);
            }

            OneHotInts = new uint[32];
            uint uintBit = 1;
            for (var i = 0; i < 32; i++)
            {
                OneHotInts[i] = uintBit << i;
            }

            OneHotLongs = new ulong[64];
            ulong ulongBit = 1;
            for (var i = 0; i < 64; i++)
            {
                OneHotLongs[i] = ulongBit << i;
            }
        }

        public static byte[] OneHotBytes;

        public static uint[] OneHotInts;

        public static ulong[] OneHotLongs;


        public static byte BitFlip(this byte bits, uint pos)
        {
            return (byte) (bits ^ OneHotBytes[pos]);
        }

        public static uint BitFlip(this uint bits, uint pos)
        {
            return bits ^ OneHotInts[pos];
        }

        public static ulong BitFlip(this ulong bits, uint pos)
        {
            return bits ^ OneHotLongs[pos];
        }


        public static IEnumerable<byte> Mutate(this IEnumerable<byte> bits, IRando randy, double mutationRate)
        {
            const int bitsLen = 8;
            return bits.Select(b =>
            {
                var bp = b;
                for (uint i = 0; i < bitsLen; i++)
                {
                    if (randy.NextDouble() < mutationRate)
                    {
                        bp = bp.BitFlip(i);
                    }
                }
                return bp;
            });
        }

        public static IEnumerable<uint> Mutate(this IEnumerable<uint> bits, IRando randy, double mutationRate)
        {
            const int bitsLen = 32;
            return bits.Select(b =>
            {
                var bp = b;
                for (uint i = 0; i < bitsLen; i++)
                {
                    if (randy.NextDouble() < mutationRate)
                    {
                        bp = bp.BitFlip(i);
                    }
                }
                return bp;
            });
        }


        public static IEnumerable<ulong> Mutate(this IEnumerable<ulong> bits, IRando randy, double mutationRate)
        {
            const int bitsLen = 64;
            return bits.Select(b =>
            {
                var bp = b;
                for (uint i = 0; i < bitsLen; i++)
                {
                    if (randy.NextDouble() < mutationRate)
                    {
                        bp = bp.BitFlip(i);
                    }
                }
                return bp;
            });
        }


        public static int[] ToIntArray(this byte bits)
        {
            var aRet = new int[8];
            byte mask = 1;

            for (var i = 0; i < 8; i++)
            {
                var masked = mask & bits;
                var sw = masked ^ mask;
                aRet[i] = (sw == 0) ? 1 : 0;
                mask <<= 1;
            }
            return aRet;
        }

        public static int[] ToIntArray(this uint bits)
        {

            var aRet = new int[32];
            uint mask = 1;

            for (var i = 0; i < 32; i++)
            {
                var masked = mask & bits;
                var sw = masked ^ mask;
                aRet[i] = (sw == 0) ? 1 : 0;
                mask <<= 1;
            }
            return aRet;
        }

        public static int[] ToIntArray(this ulong bits)
        {

            var aRet = new int[64];
            uint mask = 1;

            for (var i = 0; i < 64; i++)
            {
                var masked = mask & bits;
                var sw = masked ^ mask;
                aRet[i] = (sw == 0) ? 1 : 0;
                mask <<= 1;
            }
            return aRet;
        }


        public static uint HotCount(this byte bits)
        {
            byte mask = 1;
            uint totRet = 0;
            for (var i = 0; i < 8; i++)
            {
                var masked = mask & bits;
                var sw = masked ^ mask;
                if (sw == 0) totRet++;
                mask <<= 1;
            }
            return totRet;
        }

        public static uint HotCount(this uint bits)
        {
            uint totRet = 0;
            for (var i = 0; i < 32; i++)
            {
                var masked = OneHotInts[i] & bits;
                var sw = masked ^ OneHotInts[i];
                if (sw == 0) totRet++;
            }
            return totRet;
        }

        public static uint HotCount(this ulong bits)
        {
            uint totRet = 0;
            for (var i = 0; i < 64; i++)
            {
                var masked = OneHotInts[i] & bits;
                var sw = masked ^ OneHotInts[i];
                if (sw == 0) totRet++;
            }
            return totRet;
        }


        public static uint BitOverlap(this byte a, byte b)
        {
            return 8 - HotCount((byte)(a ^ b));
        }

        public static IEnumerable<uint> BitOverlaps(this byte[] a, byte[] b)
        {
            return a.Zip(b, (s, t) => s.BitOverlap(t));
        }

        public static IEnumerable<uint> BitOverlaps(this byte[] a, byte b)
        {
            return a.Select(s => s.BitOverlap(b));
        }



        public static uint BitOverlap(this uint a, uint b)
        {
            return 32 - HotCount(a ^ b);
        }

        public static IEnumerable<uint> BitOverlaps(this uint[] a, uint[] b)
        {
            return a.Zip(b, (s, t) => s.BitOverlap(t));
        }

        public static IEnumerable<uint> BitOverlaps(this uint[] a, uint b)
        {
            return a.Select(s => (uint)s.BitOverlap(b));
        }



        public static uint BitOverlap(this ulong a, ulong b)
        {
            return 64 - HotCount((a ^ b));
        }

        public static IEnumerable<uint> BitOverlaps(this ulong[] a, ulong[] b)
        {
            return a.Zip(b, (s, t) => s.BitOverlap(t));
        }

        public static IEnumerable<uint> BitOverlaps(this ulong[] a, ulong b)
        {
            return a.Select(s => s.BitOverlap(b));
        }



        public static int TryBitConverter()
        {
            byte[] bytes = { 0, 0, 0, 25 };
            // If the system architecture is little-endian (that is, little end first),
            // reverse the byte array.
            if (BitConverter.IsLittleEndian)
                Array.Reverse(bytes);

            var i = BitConverter.ToInt32(bytes, 0);
            var j = BitConverter.ToUInt32(bytes, 0);
            return i;
        }

    }
}
