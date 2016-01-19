// This files contains the tools used to compute the various coefficients needed
// for sampling estimates, namely the y_s, c_s, and c_st coefficients. For more
// information, see http://www.vldb.org/pvldb/vol6/p1798-nirkhiwale.pdf.

// Resources:
//   armadillo: matrix and vector containers.


// The bitwise-SWAR algorithm for computing the Hamming Weight of an integer.
// The return type is a signed integer but the actual result is an unsigned
// integer. This causes an implicit static cast, which is safe as the result is
// at most 32 and necessary because the expressions used to compute the various
// coefficients would otherwise be erroneously promoted to unsigned integers
// when they need to support negative values.
inline int CountBits(uint32_t i) {
   i = i - ((i >> 1) & 0x55555555);
   i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
   return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

// This function computes the value of c_s given s as a bit mask. It should be
// noted that when t is a subset of s, |t| + |s| = |s \ t| % 2 because whichever
// elements that appear in t will appear in s as well. This means that such an
// element will be counted twice on the left hand side and therefore not affect
// the parity of the result. Furthermore, the VLDb paper has a typo in it and t
// should in fact only range across the subsets of s and not the power set of n.
inline double ComputeCCoefficient(uint32_t s, const arma::vec& b) {
  double coef = 0;
  for (uint32_t t = 0; t <= s; t++)
    if (t & ~s)  // t is not a subset of s.
      continue;
    else
      coef += (1 - 2 * (CountBits(s - t) % 2)) * b[t];
  return coef;
}

// This function computes the value of c_s,t given s and t as bit masks. Note
// that the computation differs slightly from that described in the VLDb paper,
// which has a typo in the exponent. The power of -1 should be |t \ u| and not
// |u| + |s|, the former of which can be simplified to |t - u| because u is
// taken to be a subset of t.
inline double ComputeCCoefficient(uint32_t s, uint32_t t, const arma::vec& b) {
  double coef = 0;
  for (uint32_t u = 0; u <= t; u++)  // A bit mask representing the set u.
    if (u & ~t)  // u is not a subset of t.
      continue;
    else
      coef += (1 - 2 * (CountBits(t - u) % 2)) * b[s | u];
  return coef;
}

// These functions precompute all the necessary coefficients and store them.
template <uint32_t N>
inline void ComputeCoefficients(arma::mat::fixed<N, N>& c_st,
                                const arma::vec::fixed<N>& b) {
  for (uint32_t s = 0; s < N; s++)
    for (uint32_t t = 0; t < N; t++)
      if (s & t)  // t is not a subset of the complement of s.
        continue;
      else
        c_st(s, t) = ComputeCCoefficient(s, t, b);
}

template <uint32_t N>
inline void ComputeCoefficients(arma::vec::fixed<N>& c_s,
                                const arma::vec::fixed<N>& b) {
  for (uint32_t s = 0; s < N; s++)
    c_s(s) = ComputeCCoefficient(s, b);
}

// This computes the unbiased y_s coefficients in place. Because the mask s is
// decreasing during the loop, it is guaranteed that y[s | t] has already been
// unbiased, as s | t >= s regardless of s and t.
template <uint32_t N>
inline void UnbiasCoefficients(arma::vec::fixed<N>& y_s,
                               const arma::mat::fixed<N, N>& c_st,
                               const arma::vec::fixed<N>& b) {
  for (uint32_t s = N; s < N; s--) {
    for (uint32_t t = 1; t < N; t++)
      if (t & s)  // t is not a subset of the complement of s.
        continue;
      else
        y_s(s) -= c_st(s, t) * y_s[s | t];
    y_s(s) /= ComputeCCoefficient(s, 0);
  }
}
