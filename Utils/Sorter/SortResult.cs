﻿using Utils.Sortable;

namespace Utils.Sorter
{
    public class SortResult
    {
        public SortResult(ISorter sorter, double sortedness, bool[] stageUse,
            ISortable sortable, IPermutation result)
        {
            Sortedness = sortedness;
            StageUse = stageUse;
            Sortable = sortable;
            Result = result;
            Sorter = sorter;
        }

        public double Sortedness { get; }

        public bool[] StageUse { get; }

        public ISortable Sortable { get; }

        public ISorter Sorter { get; }

        public IPermutation Result { get; }

    }
}