using System.Collections.Generic;
using Utils.Genome;
using Utils.Genome.Sorter;

namespace Utils.Ga.Parts
{
    public static class GaDataSorterGenome
    {

        #region DimerGenomePool

        public static string kDimerGenomePool = "kDimerGenomePool";

        public static GenomePool<GenomeSorterStageDimer> GetDimerGenomePool(this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeSorterStageDimer>) dictionary[kDimerGenomePool];
        }

        public static void SetDimerGenomePool(this Dictionary<string, object> dictionary, 
            GenomePool<GenomeSorterStageDimer> sorterGenomeDimerPool)
        {
            dictionary[kDimerGenomePool] = sorterGenomeDimerPool;
        }

        #endregion


        #region BestDimerGenomePool

        public static string kBestDimerGenomePool = "kBestDimerGenomePool";

        public static GenomePool<GenomeSorterStageDimer> GetBestDimerGenomePool(this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeSorterStageDimer>)dictionary[kBestDimerGenomePool];
        }

        public static void SetBestDimerGenomePool(
            this Dictionary<string, object> dictionary,
            GenomePool<GenomeSorterStageDimer> bestDimerGenomePool)
        {
            dictionary[kBestDimerGenomePool] = bestDimerGenomePool;
        }

        #endregion


        #region ConjOrbitGenomePool

        public static string kConjOrbitGenomePool = "kConjOrbitGenomePool";

        public static GenomePool<GenomeSorterConjOrbit> GetConjOrbitGenomePool(
            this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeSorterConjOrbit>)dictionary[kConjOrbitGenomePool];
        }

        public static void SetConjOrbitGenomePool(this Dictionary<string, object> dictionary,
            GenomePool<GenomeSorterConjOrbit> sorterGenomeDimerPool)
        {
            dictionary[kConjOrbitGenomePool] = sorterGenomeDimerPool;
        }

        #endregion


        #region BestConjOrbitGenomePool

        public static string kBestConjOrbitGenomePool = "kBestConjOrbitGenomePool";

        public static GenomePool<GenomeSorterConjOrbit> GetBestConjOrbitGenomePool(
            this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeSorterConjOrbit>)dictionary[kBestConjOrbitGenomePool];
        }

        public static void SetBestConjOrbitGenomePool(this Dictionary<string, object> dictionary,
            GenomePool<GenomeSorterConjOrbit> bestConjOrbitGenomePool)
        {
            dictionary[kBestConjOrbitGenomePool] = bestConjOrbitGenomePool;
        }

        #endregion


    }
}