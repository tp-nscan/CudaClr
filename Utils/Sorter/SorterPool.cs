using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Utils.Sorter
{
    public class SorterPool : IEnumerable<ISorter>
    {
        public SorterPool(Guid id, IEnumerable<ISorter> sorters)
        {
            Id = id;
            Sorters = sorters.ToDictionary(r => r.Id);
            Order = Sorters.Values.First().SorterStages.First().Order;
        }

        public Dictionary<Guid, ISorter> Sorters { get; }

        public Guid Id { get; }

        public IEnumerator<ISorter> GetEnumerator()
        {
            return Sorters.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public uint Order { get; }

        public ISorter this[int index] => Sorters.Values.ElementAt(index);
    }


    public static class SorterPoolExt
    {
        public static SorterPool ToRandomSorterPool(this IRando rando, uint order, 
                                                  uint stageCount, uint sorterCount)
        {
            return new SorterPool(
                Guid.NewGuid(),
                0u.CountUp(sorterCount)
                          .Select(i=>rando.ToSorter(order, stageCount))
                );
        }

        public static Dictionary<ISorter, int> MakeSorterDictionary()
        {
            return new Dictionary<ISorter, int>(new SorterEqualityComparer());
        }

        public static Dictionary<ISorter, int> ToSorterDistr(this SorterPool sorterPool)
        {
            var sd = MakeSorterDictionary();

            foreach (var sorter in sorterPool.Sorters.Values)
            {
                if (sd.ContainsKey(sorter))
                {
                    sd[sorter]++;
                }
                else
                {
                    sd.Add(sorter, 1);
                }
            }

            return sd;
        }
    }
}