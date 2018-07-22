using Utils;

namespace Sponge.M2
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
        public static SimGrid<int> TestInt()
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: 128,
                height: 128,
                data: Utils.IntArrayGen.Ring(20, 46, 128, 24).ToIntBits());
        }

        public static SimGrid<int> TestInt2()
        {
            return new SimGrid<int>(
                name: "TestInt",
                width: 1024,
                height: 1024,
                data: Utils.IntArrayGen.RandInts2(1233, 1024 * 1024, 0.53).ToIntBits());
        }
    }

}
