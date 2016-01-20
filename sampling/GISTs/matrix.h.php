<?
// This GIST computes a covariance matrix for a GUS query plan involving many
// independent branches. Specifically, this query plan is given as two branches,
// one interior and one exterior. The exterior branch is then split into parts
// by a given grouping expression, similar to the splitting of a Group By. It is
// then simulated that each of these exterior sub-branches are pairwise selected
// and the covariance is computed for the two branches in a pair once they are
// joined with the interior branch. The covariance matrix is return element by
// element, with two indices and their covariance returned.

// The input to this GIST must be a GroupBy whose input is the two branches that
// have been joined and whose grouping attribute is the one described above. The
// GroupBy should have performed an adjustable Bernoulli sample and a summation
// of the value whose total sum is being estimated.

// During initialization, the minimum of the various probabilities used in the
// adjustable samples is computed.

// The first round then dispatches one worker per group and refilters the data
// using the smallest probability. This strategy introduces the issue concerning
// sample size - although each sample is initially adjusted to be between 100K
// and 400K elements, refiltering them may bring them below this range. However,
// all sub-samples must use the same probability, each sub-sample needs to be
// small to reduce computation, and only the minimum of all probabilities can be
// used because the most strict sample cannot have its probability increased
// because it already discarded the tuples that failed to pass.

// In the second round, the various Y_s coefficients are computed for the sub-
// samples with one worker per group that computes all the terms for said group.
// The terms are stored in an array with the index being the integer cast of the
// bit mask S.

// The output is then returned using a fragmented result. The covariance matrix
// is partitioned into blocks, with one fragment per block.

// Resources:
// armadillo: vectors
// math.h: pow, min
// unordered_map: unordered_map
// gist.h: HaltingGLA, SimpleScheduler
function Covariance_Matrix($t_args, $outputs, $states)
{
    // Class name is randomly generated.
    $className = generate_name('Covariance_Matrix');

    // Processing of template arguments.
    $a = $t_args['a'];
    $b = $t_args['b'];

    $numCoefs     = count($b);
    $numRelations = log($numCoefs, 2);
    $bElems       = implode(', ', $b);

    // Processing of input states.
    $states_  = array_combine(['state'], $states);
    $sampling = $states_['state']->get('inner_gla');
    $key = array_get_index($states_['state']->template_args()['group'], 0)->type();

    // Processing of outputs.
    $outputs_ = ['f' => $key, 'g' => $key, 'cov' => lookupType('float')];
    $outputs = array_combine(array_keys($outputs), $outputs_);

    $sys_headers  = ['armadillo', 'math.h', 'unordered_map'];
    $user_headers = [];
    $lib_headers  = ['base\gist.h'];
    $libraries    = [];
    $extra        = [];
    $result_type  = ['fragment'];
?>

using namespace std;
using namespace arma;

class <?=$className?>;

class <?=$className?> {
 public:
  // The various components for the GIST and Fragment result type.
  using cGLA = AnswerGLA;
  using Task = uint32_t;
  using LocalScheduler = SimpleScheduler<Task>;
  using WorkUnit = std::pair<LocalScheduler*, cGLA*>;
  using WorkUnits = std::vector<WorkUnit>;
  using Iterator = std::tuple<int, int, bool>;

  // The GLA and container used for the sub-sampling.
  using SamplingGLA = <?=$sampling?>;
  using Sample = SamplingGLA::Sample;

  // The types for the grouping and aggregated values.
  using Key = <?=$key?>;
  using Value = SamplingGLA::Value;

  // The type of mapping used to compute each coefficient.
  using Map = std::unordered_map<uint64_t, Value>;

  // The number of relations in the shared branch.
  static const constexpr int kNumRelations = <?=$numRelations?>;

  // The number of coefficients, 2 ^ kNumRelations.
  static const constexpr int kNumCoefs = <?=$numCoefs?>;

 private:
  // The GUS coefficients.
  double a;
  arma::vec::fixed<kNumCoefs> b;

  // The data structures used to hold the precomputed c coefficients.
  arma::mat::fixed<kNumCoefs, kNumCoefs> c_st;
  arma::vec::fixed<kNumCoefs> c_s;

  // The minimum probability across all sub-samples.
  double p;

  // The current iteration of the GIST.
  int round;

  // The number of groups.
  int num_groups;

  // The information needed to compute the pairwise Y_s coefficients.
  std::vector<std::array<Map, kNumCoefs>> sums;

  // The information about groups is copied from the mapping to ease lookup.
  std::vector<Key> keys;
  std::vector<const Sample*> sub_samples;

  // The container for the final results of the resampling.
  std::vector<Sample> samples;

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : a(<?=$a?>),
        b({<?=$bElems?>}),
        p(1),
        round(0),
        num_groups(state.GetMap().size()),
        sums(num_groups),
        keys(num_groups),
        sub_samples(num_groups),
        samples(num_groups) {
    // Traversing the sub-samples and computing the minimum probability.
    int index = 0;
    for (auto it = state.GetMap().begin(); it != state.GetMap().end(); ++it) {
      p = min(p, it->second.GetProbability());
      keys[index] = it->first.GetKey0();
      sub_samples[index++] = &it->second.GetSample();
    }
    cout << "min probability: " << p << endl;
    // Adjust the GUS coefficients based on the Bernoulli sub-sample.
    for (uint32_t index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 - (double) CountBits(index) / kNumRelations);
    a *= p;
    // Precomputing the c coefficients.
    for (uint32_t s = 0; s < kNumCoefs; s++) {
      c_s(s) = ComputeCCoefficient(s);
      for (uint32_t t = 0; t < kNumCoefs; t++)
        if (t & s)
          continue;
        else
          c_st(s, t) = ComputeCCoefficient(s, t);
    }
    // This is done to account for the offset when computing the covariance. Now
    // the covariance is simply the dot product of the c and y coefficients.
    c_s(0) -= a * a;
    cout << "c_s: " << c_s.t();
    cout << "c_s,t: " << c_st;
  }

  // In both rounds, one worker per group is allocated.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    round++;
    for (uint32_t counter = 0; counter < num_groups; counter++)
      workers.push_back(WorkUnit(new LocalScheduler(counter),
                                 new cGLA(round == 1)));
  }

  // In the first round
  void DoStep(Task& task, cGLA& gla) {
    if (round == 1) {
      SamplingGLA::Refilter(samples[task], *sub_samples[task], p);
      cout << "sample " << task << " size: " << samples[task].size() << endl;
    } else
      // The various sums need for the final coefficients are computed.
      for (uint32_t mask = 0; mask < kNumCoefs; mask++) {
        for (auto item : samples[task])
          Update(sums[mask][task], ChainHash(item.first, task), item.second);
        cout << "task: " << task << " mask: " << mask << " size: " << sums[mask][task].size() << endl;
      }
  }

  // One fragment per distinct pair is allocated, including pairs of the same
  // group because their covariance is then the variance and not trivial.
  int GetNumFragments() {
    return num_groups * (num_groups + 1) / 2;
  }

  // The iterator is triplet consisting of the two integers identifying the two
  // groups in the pair and a boolean that marks whether this fragment is done.
  Iterator* Finalize(int fragment) {
    // The co-ordinates of the current fragment in the matrix grid are computed.
    int col = (sqrt(1 + 8 * fragment) - 1) / 2;
    int row = fragment - col * (col + 1) / 2;
    return new Iterator(row, col, false);
  }

  // This is the step that computes the covariance for the pair.
  bool GetNextResult(Iterator* it, <?=typed_ref_args($outputs_)?>) {
    if (std::get<2>(*it))  // The result has already been returned.
      return false;
    // This container holds both biased and unbiased coefficients.
    arma::vec::fixed<kNumCoefs> y(arma::fill::zeros);
    // Computing the pairwise product.
    for (uint32_t mask = 0; mask < kNumCoefs; mask++) {
      auto& map_1 = sums[mask][get<0>(*it)];
      auto& map_2 = sums[mask][get<1>(*it)];
      for (auto item_1 : map_1) {
        auto item_2 = map_2.find(item_1.first);
        if (item_2 != map_2.end())
          y(mask) += item_1.second * item_2->second;
      }
    }
    // cout << get<0>(*it) << " " << get<1>(*it) << " biased: " << y.t();
    // Unbiasing the Y_s coefficients.
    for (uint32_t s = kNumCoefs - 1; s < kNumCoefs; s--) {
      for (uint32_t t = 1; t <= kNumCoefs - 1; t++) {
        if (t & s)  // t is not a subset of the complement of s.
          continue;
        y(s) -= ComputeCCoefficient(s, t) * y[s | t];
      }
      y(s) /= ComputeCCoefficient(s, 0);
    }
    // cout << get<0>(*it) << " " << get<1>(*it) << " unbiased: " << y.t();
    // Returning the values.
    f = keys[std::get<0>(*it)];
    g = keys[std::get<1>(*it)];
    cov = dot(y, c_s);
    std::get<2>(*it) = true;
    return true;
  }

 private:
  // This updates a map given a value and key.
  void Update(Map& map, uint64_t key, Value value) {
    Map::iterator it = map.find(key);
    if (it == map.end()) {
      auto insertion = map.insert(Map::value_type(key, 0));
      it = insertion.first;
    }
    it->second += value;
  }

  // This function takes in a set of keys and a bit mask specifying which keys
  // are to be used. It hashes each relevent key to a uint64_t and then uses a
  // chain hash to condense these hashed values into a single output. The mask
  // is used as the starting value for the chain hash.
  uint64_t ChainHash(SamplingGLA::KeySet keys, uint32_t mask) {
    uint64_t result = mask;
<?  for ($index = 0; $index < $numRelations; $index++) { ?>
    if (mask & 1 << <?=$index?>)
        result = CongruentHash(result, Hash(get<<?=$index?>>(keys)));
<?  } ?>
    return result;
  }

  // This function computes the value of c_s,t given s and t as bit masks.
  double ComputeCCoefficient(uint32_t s, uint32_t t) {
    double coef = 0;
    for (uint32_t u = 0; u <= t; u++) {  // A bit mask representing the set u.
      if (u & ~t)  // u is not a subset of t and we ignore this case.
        continue;
      // This differs from the vldb paper which says the exponent is |s| + |u|.
      // This is intentional as that is a typo. It should read |t \ u| which is
      // equivalent to |t - u|, as u is a subset of t.
      coef += (1 - 2 * (int) (CountBits(t - u) % 2)) * b[s | u];
    }
    return coef;
  }

  // This function computes the value of c_s given s as a bit mask.
  double ComputeCCoefficient(uint32_t s) {
    double coef = 0;
    for (uint32_t t = 0; t <= s; t++) {
      if (t & ~s)  // t is not a subset of s.
        continue;
      // Because t is a subset of s, |t| + |s| = |s \ t| mod 2 because whichever
      // element appears in t will appear in s as well. This means that such an
      // element will be counted twice on the left hand side and therefore not
      // affect the parity of the result.
      coef += (1 - 2 * (int) (CountBits(s - t) % 2)) * b[t];
    }
    return coef;
  }

  // The bitwise-SWAR algorithm for computing the Hamming Weight of an integer.
  uint32_t CountBits(uint32_t i) {
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
  }
};

<?
    return [
        'kind'            => 'GIST',
        'name'            => $className,
        'system_headers'  => $sys_headers,
        'user_headers'    => $user_headers,
        'lib_headers'     => $lib_headers,
        'libraries'       => $libraries,
        'extra'           => $extra,
        'iterable'        => true,
        'intermediate'    => false,
        'output'          => $outputs,
        'result_type'     => $result_type,
    ];
}
?>
