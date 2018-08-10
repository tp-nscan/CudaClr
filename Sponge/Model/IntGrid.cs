using System.Collections;
using System.Linq;
using Utils;

namespace Sponge.Model
{
    public class SimGrid<T>
    {
        public SimGrid(string name, uint width, uint height, T[] data)
        {
            Name = name;
            Width = width;
            Height = height;
            Data = data;
        }

        string _name;
        public string Name
        {
            get { return _name; }
            private set { _name = value; }
        }

        uint _width;
        public uint Width
        {
            get { return _width; }
            private set { _width = value; }
        }

        uint _height;
        public uint Height
        {
            get { return _height; }
            private set { _height = value; }
        }

        T[] _data;
        public T[] Data
        {
            get { return _data; }
            private set { _data = value; }
        }

        public uint Area { get { return Width * Height; } }

    }

    public static class SimGridSamples
    {
        public static SimGrid<int> SquareRingBits(uint span)
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: span,
                height: span,
                data: Utils.IntArrayGen.Ring(span/6, span/3, span, 2).ToIntBits());
        }

        public static SimGrid<int> SquareRandBits(uint span, int seed)
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: span,
                height: span,
                data: Utils.IntArrayGen.RandInts2(seed, span * span, 0.5).ToIntBits());
        }

        public static SimGrid<int> UniformVals(uint span, int val)
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: span,
                height: span,
                data: Enumerable.Repeat(val, (int)(span* span)).ToArray());
        }

    }

}
