using System.Linq;

namespace Utils
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

    public static class SimGridIntSamples
    {
        public static SimGrid<int> SquareRingBits(uint span)
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: span,
                height: span,
                data: IntArrayGen.Ring(span/6, span/3, span, 2).ToIntBits());
        }

        public static SimGrid<int> SquareRandBits(uint span, int seed)
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: span,
                height: span,
                data: IntArrayGen.RandInts2(seed, span * span, 0.5).ToIntBits());
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

    public static class SimGridFloatSamples
    {
        public static SimGrid<float> LeftRightGradient(uint span)
        {
            return new SimGrid<float>(
                name: "LeftRightGradientFloats",
                width: span,
                height: span,
                data: FloatArrayGen.LeftRightGradient(
                    span:span, 
                    low_val: 0.0f, 
                    high_val:1.0f));
        }

        public static SimGrid<float> RandUniform0_1(uint span, int seed)
        {
            return new SimGrid<float>(
                name: "RandUniform0_1",
                width: span,
                height: span,
                data: FloatArrayGen.UnSignedUnitUniformRands(
                    span: span,
                    seed: seed));
        }

    }

}
