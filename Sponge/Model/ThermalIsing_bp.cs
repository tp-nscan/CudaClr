using System;
using System.Linq;
using RandoClr;
using System.Diagnostics;
using System.Collections.Generic;
using CuArrayClr;
using GridProcsClr;
using Utils;

namespace Sponge.Model
{
    public static class ThermalIsing_bp
    {
        public const int _tempSteps = 256;
        public const int _allTempSteps = _tempSteps + 1;

        private static IntPtr d_flipData;
        private static IntPtr d_tempData;
        private static IntPtr d_indexRands;
        private static IntPtr d_flipRands;
        private static IntPtr d_threshes;
        private static IntPtr d_heatBlocks;

        private static uint _span;
        private static uint _block_size;
        private static uint _blocks_per_span;
        private static uint _area;
        private static uint _blockCount;
        private static CudaArray _cudaArray;
        private static GridProcs _gridProcs;
        private static RandoProcs _randoProcs;
        private static Stopwatch _stopwatch = new Stopwatch();

        public static string Init(float[] temp_inputs, int[] flip_inputs, uint span, uint blockSize, int seed)
        {
            _span = span;
            _block_size = blockSize;

            _area = _span * _span;
            _blocks_per_span = span / blockSize;
            _blockCount = _blocks_per_span * _blocks_per_span;

            d_flipData = new IntPtr();
            d_tempData = new IntPtr();

            d_flipRands = new IntPtr();
            d_indexRands = new IntPtr();
            d_threshes = new IntPtr();

            _cudaArray = new CudaArray();
            _gridProcs = new GridProcs();
            _randoProcs = new RandoProcs();

            var strRet = _cudaArray.ResetDevice();
            strRet = strRet + _randoProcs.MakeGenerator32(seed);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_tempData, _area);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_heatBlocks, _area / 1024);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_flipData, _area);

            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_indexRands, _blockCount);
            strRet = strRet + _cudaArray.MallocIntsOnDevice(ref d_flipRands, _blockCount);
            strRet = strRet + _cudaArray.MallocFloatsOnDevice(ref d_threshes, _allTempSteps);

            strRet = strRet + _cudaArray.CopyIntsToDevice(flip_inputs, d_flipData, _area);
            strRet = strRet + _cudaArray.CopyFloatsToDevice(temp_inputs, d_tempData, _area);


            var res9 = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(res9, d_flipData, _area);

            return strRet;
        }

        public static ProcResult UpdateH(int steps, float qRate, float filpEnergy, float beta)
        {
            var strRet = String.Empty;

            _stopwatch.Reset();
            _stopwatch.Start();

            for (var s = 0; s < steps; s++)
            {
                var res9 = new int[_area];
                strRet = strRet + _cudaArray.CopyIntsFromDevice(res9, d_flipData, _area);

                var bbs = FloatFuncs.Betas(_tempSteps, beta);
                strRet = strRet + _cudaArray.CopyFloatsToDevice(bbs, d_threshes, _allTempSteps);
                strRet = strRet + _randoProcs.MakeRandomInts(d_indexRands, _blockCount);
                strRet = strRet + _randoProcs.MakeUniformRands(d_flipRands, _blockCount);

                strRet = strRet + _gridProcs.Run_k_ThermoIsing_bp(
                        temp_data: d_tempData,
                        flip_data: d_flipData,
                        index_rands: d_indexRands,
                        flip_rands: d_flipRands,
                        threshes: d_threshes,
                        flip_energy: filpEnergy,
                        block_size: _block_size,
                        blocks_per_span: _blocks_per_span,
                        q_rate: qRate);
            }

            _stopwatch.Stop();

            var dRet = new Dictionary<string, object>();


            float[] bhts = new float[_area / 1024];
            strRet = strRet + _cudaArray.RunBlockAddFloats_32_Kernel(
                                    destPtr: d_heatBlocks,
                                    srcPtr: d_tempData,
                                    span: _span
                );

            strRet = strRet + _cudaArray.CopyFloatsFromDevice(bhts, d_heatBlocks, _area / 1024);
            float tot = bhts.Sum();
            tot /= _area;
            dRet["TotalHeat"] = tot;


            var tres = new float[_area];
            strRet = strRet + _cudaArray.CopyFloatsFromDevice(tres, d_tempData, _area);
            dRet["ThermGrid"] = new SimGrid<float>(name: "Therms",
                                                   width: _span,
                                                   height: _span,
                                                   data: tres);


            var fres = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(fres, d_flipData, _area);
            dRet["FlipGrid"] = new SimGrid<int>(name: "Flips",
                                                width: _span,
                                                height: _span,
                                                data: fres);


            return new ProcResult(data: dRet,
                                  err: strRet,
                                  steps: steps,
                                  time: _stopwatch.ElapsedMilliseconds);
        }


        public static ProcResult UpdateHf(int steps, float qRate, float filpEnergy, float beta)
        {
            var strRet = String.Empty;

            _stopwatch.Reset();
            _stopwatch.Start();

            for (var s = 0; s < steps; s++)
            {
                var res9 = new int[_area];
                strRet = strRet + _cudaArray.CopyIntsFromDevice(res9, d_flipData, _area);

                var bbs = FloatFuncs.Betas(_tempSteps, beta);
                strRet = strRet + _cudaArray.CopyFloatsToDevice(bbs, d_threshes, _allTempSteps);
                strRet = strRet + _randoProcs.MakeRandomInts(d_indexRands, _blockCount);
                strRet = strRet + _randoProcs.MakeUniformRands(d_flipRands, _blockCount);

                strRet = strRet + _gridProcs.Run_k_ThermoIsing_bp(
                        temp_data: d_tempData,
                        flip_data: d_flipData,
                        index_rands: d_indexRands,
                        flip_rands: d_flipRands,
                        threshes: d_threshes,
                        flip_energy: filpEnergy,
                        block_size: _block_size,
                        blocks_per_span: _blocks_per_span,
                        q_rate: qRate);
            }

            _stopwatch.Stop();

            var dRet = new Dictionary<string, object>();


            float[] bhts = new float[_area / 1024];
            strRet = strRet + _cudaArray.RunBlockAddFloats_32_Kernel(
                                    destPtr: d_heatBlocks,
                                    srcPtr: d_tempData,
                                    span: _span
                );

            strRet = strRet + _cudaArray.CopyFloatsFromDevice(bhts, d_heatBlocks, _area / 1024);
            float tot = bhts.Sum();
            tot /= _area;
            dRet["TotalHeat"] = tot;


            var tres = new float[_area];
            strRet = strRet + _cudaArray.CopyFloatsFromDevice(tres, d_tempData, _area);
            dRet["ThermGrid"] = new SimGrid<float>(name: "Therms",
                                                   width: _span,
                                                   height: _span,
                                                   data: tres);


            var fres = new int[_area];
            strRet = strRet + _cudaArray.CopyIntsFromDevice(fres, d_flipData, _area);
            dRet["FlipGrid"] = new SimGrid<int>(name: "Flips",
                                                width: _span,
                                                height: _span,
                                                data: fres);


            return new ProcResult(data: dRet,
                                  err: strRet,
                                  steps: steps,
                                  time: _stopwatch.ElapsedMilliseconds);
        }

    }
}
