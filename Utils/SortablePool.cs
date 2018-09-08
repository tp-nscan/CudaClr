using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public class SortablePool : IEnumerable<ISortable>
    {
        public SortablePool(Guid id, IEnumerable<ISortable> sortables)
        {
            Id = id;
            Sortables = sortables.ToDictionary(r => r.Id);
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

        public ISortable this[int index] => Sortables.Values.ElementAt(index);

    }

    public static class SortablePoolExt
    {
        public static SortablePool RandomSortablePool(this IRando rando, int order, int sortableCount)
        {
            return new SortablePool(Guid.NewGuid(),
                Enumerable.Range(0, sortableCount).Select(i=>
                rando.RandomPermutation(order).ToSortable()));
        }

        public static SortablePool OrbitSortablePool(IPermutation seed, int maxSize)
        {
            return new SortablePool(Guid.NewGuid(),
                seed.GetOrbit(maxSize).Keys.Select(p => p.ToSortable()));
        }

        public static SorterResult Sort(this SortablePool sortablePool, ISorter sorter)
        {
            return new SorterResult(sortablePool.Select(sorter.Sort), false);
        }

       // public static 
    }

}