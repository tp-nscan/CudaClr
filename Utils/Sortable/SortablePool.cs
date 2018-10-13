using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Utils.Sorter;

namespace Utils.Sortable
{
    public class SortablePool : IEnumerable<ISortable>
    {
        public SortablePool(Guid id, IEnumerable<ISortable> sortables)
        {
            Id = id;
            Sortables = sortables.ToDictionary(r => r.Id);
            Order = Sortables.Values.First() .Order;
        }

        public Dictionary<Guid, ISortable> Sortables { get; }

        public Guid Id { get; }

        public IEnumerator<ISortable> GetEnumerator()
        {
            return Sortables.Values.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }

        public uint Order { get; }

        public ISortable this[int index] => Sortables.Values.ElementAt(index);

    }

    public static class SortablePoolExt
    {
        public static SortablePool ToRandomSortablePool(this IRando rando, uint order, uint poolCount)
        {
            return new SortablePool(Guid.NewGuid(),
                0u.CountUp(poolCount).Select(i=>
                rando.ToPermutation(order).ToSortable()));
        }

        public static SortablePool OrbitSortablePool(IPermutation seed, uint maxSize)
        {
            return new SortablePool(Guid.NewGuid(),
                seed.GetOrbit(maxSize).Keys.Select(p => p.ToSortable()));
        }

        public static SorterResult Sort(this SortablePool sortablePool, ISorter sorter)
        {
            return new SorterResult(sortablePool.Select(sorter.Sort), false);
        }
        
    }

}