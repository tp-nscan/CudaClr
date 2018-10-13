using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Utils
{
    public class StringFuncs
    {
        public static string GridFormat(uint rows, uint cols, Func<uint, uint, string> cellFunc)
        {
            var sb = new StringBuilder();

            for (uint i = 0; i < rows; i++)
            {
                for (uint j = 0; j < cols; j++)
                {
                    sb.Append(cellFunc(i, j));
                    sb.Append("\t");
                }

                sb.Append("\n");
            }
            return sb.ToString();
        }

        public static string LineFormat(uint[] data)
        {
            var sb = new StringBuilder();

            for (uint i = 0; i < data.Length; i++)
            {
                    sb.Append(data[i]);
                    sb.Append(" ");
            }
            return sb.ToString();
        }

    }
}
