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

        public static uint MutateAbit(this uint bits, IRando randy)
        {
            var mask = OneHotInts[randy.NextInt(32)];
            return bits ^ mask;
        }

        public static uint MutateAbit(this byte bits, IRando randy)
        {
            var mask = OneHotInts[randy.NextInt(32)];
            return bits ^ mask;
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














        public static int TryThis()
        {
            byte[] bytes = { 0, 0, 0, 25 };
            byte[] bytes2 = { 0, 0, 25, 0 };

            var bv = (bytes[3] ^ bytes2[3]);
            byte[] bytesOr = new byte[4];

            bytesOr[0] = (byte) bv;


            uint z = 25;
            uint w = 400;
            uint x = 25 ^ 400;

            //byte[] bytes3 =
            //{
            //    bytes[0] ^ bytes2[0],
            //    bytes[0] ^ bytes2[0],
            //    bytes[0] ^ bytes2[0],
            //    bytes[0] ^ bytes2[0]

            //};

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
