using System;
using System.Collections.Generic;
using System.Linq;

namespace Utils
{
    public class DiscDistrib<T>
    {
        public DiscDistrib(IEnumerable<Tuple<double, T>> bins)
        {
            var lb = bins.ToList();
            var sum = lb.Sum(b => b.Item1);

            var cume = 0.0;
            for (var i = 0; i < lb.Count; i++)
            {
                cume += lb[i].Item1 / sum;
                _bins.Add(new Tuple<double, T>(cume, lb[i].Item2));
            }
        }

        public T Draw(IRando rando)
        {
            var rv = rando.NextDouble();
            for (var i = 0; i < _bins.Count; i++)
            {
                if (_bins[i].Item1 > rv)
                {
                    return _bins[i].Item2;
                }
            }
            throw new Exception("bin not selected in DiscDistrib.Draw");
        }

        private readonly List<Tuple<double, T>> _bins 
            = new List<Tuple<double, T>>();

        private List<Tuple<double, T>> Bins => _bins;
    }
}
