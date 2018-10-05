using System.Collections.Generic;
using Utils.Genome;

namespace Utils.Ga.Parts
{
    public static class SortingGaDataGenome
    {
        #region DimerGenomePool

        public static string kDimerGenomePool = "kDimerGenomePool";

        public static GenomePool<GenomeDimer> GetDimerGenomePool(this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeDimer>) dictionary[kDimerGenomePool];
        }

        public static void SetDimerGenomePool(this Dictionary<string, object> dictionary, 
            GenomePool<GenomeDimer> sorterGenomeDimerPool)
        {
            dictionary[kDimerGenomePool] = sorterGenomeDimerPool;
        }

        #endregion


        #region BestDimerGenomePool

        public static string kBestDimerGenomePool = "kBestDimerGenomePool";

        public static GenomePool<GenomeDimer> GetBestDimerGenomePool(this Dictionary<string, object> dictionary)
        {
            return (GenomePool<GenomeDimer>)dictionary[kBestDimerGenomePool];
        }

        public static void SetBestDimerGenomePool(this Dictionary<string, object> dictionary,
            GenomePool<GenomeDimer> bestDimerGenomePool)
        {
            dictionary[kBestDimerGenomePool] = bestDimerGenomePool;
        }

        #endregion

    }
}