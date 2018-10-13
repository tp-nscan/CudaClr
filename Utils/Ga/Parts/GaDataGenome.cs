using System.Collections.Generic;
using Utils.Genome;

namespace Utils.Ga.Parts
{
    public static class GaDataGenome
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

        public static void SetBestDimerGenomePool(this Dictionary<string, object> dictionary,
            GenomePool<GenomeStageDimer> bestDimerGenomePool)
        {
            dictionary[kBestDimerGenomePool] = bestDimerGenomePool;
        }

        #endregion

    }
}