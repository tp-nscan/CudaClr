using System;
using System.Collections.Generic;

namespace Utils.Sortable
{
    public interface ISortable : IPermutation
    {
        Guid Id { get; }
    }

    public class Sortable : Permutation, ISortable
    {
        public Sortable(Guid id, uint order, IEnumerable<uint> terms) : base(order, terms)
        {
            Id = id;
        }

        public Guid Id { get; }
    }


    public static class SortableExt
    {

        public static IPermutation GetPermutation(this ISortable sortable)
        {
            return PermutationEx.MakePermutation(
                sortable.GetMap()
            );
        }

        public static ISortable ToSortable(this IPermutation perm)
        {
            return new Sortable(Guid.NewGuid(), perm.Order, perm.GetMap());
        }

        public static ISortable Mutate(this ISortable sortable, IRando rando, float replacementRate)
        {
            if (rando.NextDouble() < replacementRate)
            {
                return rando.ToPermutation(sortable.Order).ToSortable();
            }
            return sortable;
        }

    }



}