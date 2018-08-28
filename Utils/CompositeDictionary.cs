using System;
using System.Collections.Generic;

public class CompositeDictionary<TKey, TValue> : Dictionary<TKey, TValue>
{
    public CompositeDictionary(params Func<TKey, object>[] props) : base(new CompositeEqualityComparer<TKey>(props))
    {
    }
}