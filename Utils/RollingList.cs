using System.Collections;
using System.Collections.Generic;

namespace Utils
{
    public class RollingList<T> : IEnumerable<T>
    {
        public RollingList(uint capacity)
        {
            Capacity = capacity;
            _linkedList = new LinkedList<T>();
        }

        public void Add(T item)
        {
            if (_linkedList.Count == Capacity)
            {
                _linkedList.RemoveLast();
            }

            _linkedList.AddFirst(item);
        }

        private LinkedList<T> _linkedList;

        public uint Capacity { get; }

        public IEnumerator<T> GetEnumerator()
        {
            return _linkedList.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }
}
