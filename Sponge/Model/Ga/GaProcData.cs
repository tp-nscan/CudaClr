using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Utils.Ga;

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
