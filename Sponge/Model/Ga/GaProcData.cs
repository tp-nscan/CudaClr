using System.Collections.Generic;
using Utils.Ga.Parts;

namespace Sponge.Model.Ga
{
    public static class GaProcData
    {
        #region GaSortingData

        public static string kSorterWinRate = "kGaSortingData";

        public static GaSortingData GetGaSortingData(this Dictionary<string, object> dictionary)
        {
            return (GaSortingData)dictionary[kSorterWinRate];
        }

        public static void SetGaSortingData(this Dictionary<string, object> dictionary, GaSortingData sorterWinRate)
        {
            dictionary[kSorterWinRate] = sorterWinRate;
        }

        #endregion


    }
}
