﻿using System.Linq;

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

        public string Name { get; private set; }

        public uint Width { get; private set; }

        public uint Height { get; private set; }

        public T[] Data { get; private set; }

        public uint Area => Width * Height;
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

        public static SimGrid<float> HiLow(uint span, float lowVal, float hiVal)
        {
            return new SimGrid<float>(
                name: "HiLow",
                width: span,
                height: span,
                data: FloatArrayGen.HiLow(
                    span: span,
                    lowVal: lowVal,
                    hiVal: hiVal));
        }

    }

}
