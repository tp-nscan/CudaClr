using System.Collections.Generic;
using Utils.Sortable;
using Utils.Sorter;

namespace Utils.Ga.Parts
{

    public static class SortingGaDataCore
    {

        #region SorterWinRate

        public static string kSorterWinRate = "kSorterWinRate";

        public static double GetSorterWinRate(this Dictionary<string, object> dictionary)
        {
            return (double) dictionary[kSorterWinRate];
        }

        public static void SetSorterWinRate(this Dictionary<string, object> dictionary, double sorterWinRate)
        {
            dictionary[kSorterWinRate] = sorterWinRate;
        }

        #endregion


        #region SortableWinRate

        public static string kSortableWinRate = "kSortableWinRate";

        public static double GetSortableWinRate(this Dictionary<string, object> dictionary)
        {
            return (double) dictionary[kSortableWinRate];
        }

        public static void SetSortableWinRate(this Dictionary<string, object> dictionary, double sortableWinRate)
        {
            dictionary[kSortableWinRate] = sortableWinRate;
        }

        #endregion


        #region SorterPool

        public static string kSorterPool = "kSorterPool";

        public static SorterPool GetSorterPool(this Dictionary<string, object> dictionary)
        {
            return (SorterPool) dictionary[kSorterPool];
        }

        public static void SetSorterPool(this Dictionary<string, object> dictionary, SorterPool sorterPool)
        {
            dictionary[kSorterPool] = sorterPool;
        }

        #endregion


        #region SortablePool

        public static string kSortablePool = "kSortablePool";

        public static SortablePool GetSortablePool(this Dictionary<string, object> dictionary)
        {
            return (SortablePool) dictionary[kSortablePool];
        }

        public static void SetSortablePool(this Dictionary<string, object> dictionary, SortablePool sorterPool)
        {
            dictionary[kSortablePool] = sorterPool;
        }

        #endregion


        #region SortingResults

        public static string kSortingResults = "kSortingResults";

        public static SortingResults GetSortingResults(this Dictionary<string, object> dictionary)
        {
            return (SortingResults) dictionary[kSortingResults];
        }

        public static void SetSortingResults(this Dictionary<string, object> dictionary, SortingResults sorterPool)
        {
            dictionary[kSortingResults] = sorterPool;
        }

        #endregion


        #region BestSorterPool

        public static string kBestSorterPool = "kBestSorterPool";

        public static SorterPool GetBestSorterPool(this Dictionary<string, object> dictionary)
        {
            return (SorterPool) dictionary[kBestSorterPool];
        }

        public static void SetBestSorterPool(this Dictionary<string, object> dictionary, SorterPool bestSorterPool)
        {
            dictionary[kBestSorterPool] = bestSorterPool;
        }

        #endregion


        #region BestSortablePool

        public static string kBestSortablePool = "kBestSortablePool";

        public static SortablePool GetBestSortablePool(this Dictionary<string, object> dictionary)
        {
            return (SortablePool) dictionary[kBestSortablePool];
        }

        public static void SetBestSortablePool(this Dictionary<string, object> dictionary,
            SortablePool bestSortablePool)
        {
            dictionary[kBestSortablePool] = bestSortablePool;
        }

        #endregion


        #region CurrentStep

        public static string kCurrentStep = "kCurrentStep";

        public static int GetCurrentStep(this Dictionary<string, object> dictionary)
        {
            return (int) dictionary[kCurrentStep];
        }

        public static void SetCurrentStep(this Dictionary<string, object> dictionary, int currentStep)
        {
            dictionary[kCurrentStep] = currentStep;
        }

        #endregion


        #region StageReplacementMode

        public static string kStageReplacementMode = "kStageReplacementMode";

        public static StageReplacementMode GetStageReplacementMode(this Dictionary<string, object> dictionary)
        {
            return (StageReplacementMode) dictionary[kStageReplacementMode];
        }

        public static void SetStageReplacementMode(this Dictionary<string, object> dictionary,
            StageReplacementMode stageReplacementMode)
        {
            dictionary[kStageReplacementMode] = stageReplacementMode;
        }

        #endregion


    }

}