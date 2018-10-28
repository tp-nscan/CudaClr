using System.Collections.Generic;
using Utils.Ga.Parts;

namespace Sponge.Model.Ga
{
    public static class GaProcData
    {
        #region GaSortingData

        public static string kSorterWinRate = "kGaSortingData";

        public static GaData GetGaSortingData(this Dictionary<string, object> dictionary)
        {
            return (GaData)dictionary[kSorterWinRate];
        }

        public static void SetGaSortingData(this Dictionary<string, object> dictionary, GaData sorterWinRate)
        {
            dictionary[kSorterWinRate] = sorterWinRate;
        }

        #endregion


    }
}
