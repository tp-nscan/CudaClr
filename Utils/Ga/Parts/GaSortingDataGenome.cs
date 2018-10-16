using System.Collections.Generic;
using Utils.Genome;
using Utils.Genome.Sorter;

namespace Utils.Ga.Parts
{
    public static class GaSortingDataGenome
    {
        #region DimerGenomePool

        public static string kDimerGenomePool = "kDimerGenomePool";

        public static GenomePool<GenomeStageDimer> GetDimerGenomePool(this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeStageDimer>) dictionary[kDimerGenomePool];
        }

        public static void SetDimerGenomePool(this Dictionary<string, object> dictionary, 
            GenomePool<GenomeStageDimer> sorterGenomeDimerPool)
        {
            dictionary[kDimerGenomePool] = sorterGenomeDimerPool;
        }

        #endregion


        #region BestDimerGenomePool

        public static string kBestDimerGenomePool = "kBestDimerGenomePool";

        public static GenomePool<GenomeStageDimer> GetBestDimerGenomePool(this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeStageDimer>)dictionary[kBestDimerGenomePool];
        }

        public static void SetBestDimerGenomePool(
            this Dictionary<string, object> dictionary,
            GenomePool<GenomeStageDimer> bestDimerGenomePool)
        {
            dictionary[kBestDimerGenomePool] = bestDimerGenomePool;
        }

        #endregion



        #region ConjOrbitGenomePool

        public static string kConjOrbitGenomePool = "kConjOrbitGenomePool";

        public static GenomePool<GenomeConjOrbit> GetConjOrbitGenomePool(
            this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeConjOrbit>)dictionary[kConjOrbitGenomePool];
        }

        public static void SetConjOrbitGenomePool(this Dictionary<string, object> dictionary,
            GenomePool<GenomeConjOrbit> sorterGenomeDimerPool)
        {
            dictionary[kConjOrbitGenomePool] = sorterGenomeDimerPool;
        }

        #endregion


        #region BestConjOrbitGenomePool

        public static string kBestConjOrbitGenomePool = "kBestConjOrbitGenomePool";

        public static GenomePool<GenomeConjOrbit> GetBestConjOrbitGenomePool(
            this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeConjOrbit>)dictionary[kBestConjOrbitGenomePool];
        }

        public static void SetBestConjOrbitGenomePool(this Dictionary<string, object> dictionary,
            GenomePool<GenomeConjOrbit> bestConjOrbitGenomePool)
        {
            dictionary[kBestConjOrbitGenomePool] = bestConjOrbitGenomePool;
        }

        #endregion




    }
}