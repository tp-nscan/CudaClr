using System;
using System.Collections.Generic;

class CompositeEqualityComparer<T> : IEqualityComparer<T>
{
    public Func<T, object>[] Lambdas { get; set; }

    public CompositeEqualityComparer(params Func<T, object>[] propLambdas)
    {
        Lambdas = propLambdas;
    }
    public bool Equals(T obj1, T obj2)
    {
        bool isEqual = true;
        foreach (var p in Lambdas)
        {
            isEqual &= p(obj1).Equals(p(obj2));
            if (!isEqual)
                return false;
        }
        return isEqual;
    }

    public int GetHashCode(T obj)
    {
        int hCode = 0;
        foreach (var p in Lambdas)
        {
            hCode ^= p(obj).GetHashCode();
        }
        return hCode;
    }
}