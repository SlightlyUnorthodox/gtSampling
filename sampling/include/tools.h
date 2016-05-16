// This files contains various helper functions used in sampling estimates.

// Resources:
// base/HashFct.h: SpookyHash
// unordered_map: unordered_map
#include "base/HashFct.h"
#include <unordered_map>

// The template that encodes the indices used throughout these functions.
template <size_t N> struct int_{ };

// These functions take in a tuple of keys and a bit mask specifying which keys
// are to be used. It hashes each necessary key to a uint64_t and then uses a
// chain hash to condense these hashed values into a single output. The mask is
// used as the starting value for the chain hash.

// It is important to note that these functions rely on the definitions of Hash,
// which is provided automatically by the GrokIt system. These must be supplied
// if this code is to be used outside of GrokIt.

// The recursive traversal across the tuple.
template <class T, size_t N>
inline uint64_t ChainHash(const T& tuple, uint32_t mask, uint64_t seed, int_<N>) {
  if (mask & 1 << N)
    seed = SpookyHash(seed, Hash(std::get<N>(tuple)));
  return ChainHash(tuple, mask, seed, int_<N - 1>());
}

// The base case.
template <class T>
inline uint64_t ChainHash(const T& tuple, uint32_t mask, uint64_t seed, int_<0>) {
  if (mask & 1)
    seed = SpookyHash(seed, Hash(std::get<0>(tuple)));
  return seed;
}

// The helper function that sets everything up.
template <class... Args>
inline uint64_t ChainHash(const std::tuple<Args...>& t, uint32_t mask) {
  return ChainHash(t, mask, mask, int_<sizeof...(Args) - 1>());
}

// This updates a map containing the sum of values for a given key.
template <typename Map>
inline void Update(Map& map, const typename Map::key_type& key,
                   const typename Map::mapped_type& value) {
  auto it = map.find(key);
  if (it == map.end())
    map.insert(std::make_pair(key, value));
  else
    it->second += value;
}

// These are used to recursively print the tuples, a tool that is only used for
// debugging the code. Because this is a common application, these functions are
// wholly copied from http://cpplove.blogspot.com/2012/07/printing-tuples.html.
template <class Tuple, size_t Pos>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<Pos> ) {
  out << std::get< std::tuple_size<Tuple>::value-Pos >(t) << ',';
  return print_tuple(out, t, int_<Pos-1>());
}

template <class Tuple>
std::ostream& print_tuple(std::ostream& out, const Tuple& t, int_<1> ) {
  return out << std::get<std::tuple_size<Tuple>::value-1>(t);
}

template <class... Args>
std::ostream& operator<<(std::ostream& out, const std::tuple<Args...>& t) {
  out << '(';
  print_tuple(out, t, int_<sizeof...(Args)>());
  return out << ')';
}
