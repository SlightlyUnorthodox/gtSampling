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
    $foreignKey = $t_args['foreign.key'];

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
    $lib_headers  = ['base\gist.h', 'coefs.h', 'tools.h'];
    $libraries    = [];
    $extra        = [];
    $result_type  = ['multi'];
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

  // The probabilities for the various sub-samples.
  arma::vec p_vec;

  // The iterator for the multi return type.
  Iterator multi_it;

  arma::wall_clock timer;

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : a(<?=$a?>),
        b({<?=$bElems?>}),
        round(0),
        num_groups(state.GetMap().size()),
        sums(num_groups),
        keys(num_groups),
        sub_samples(num_groups),
        samples(num_groups),
        p_vec(num_groups) {
    timer.tic();
    // Traversing the sub-samples and computing the minimum probability.
    int index = 0;
    for (auto it = state.GetMap().begin(); it != state.GetMap().end(); ++it) {
      cout << "sample " << index << " size: " << it->second.GetSample().size() << endl;
      p_vec(index) = it->second.GetProbability();
      keys[index] = it->first.GetKey0();
      sub_samples[index++] = &it->second.GetSample();
    }
    p = p_vec.min();
    cout << "probabilities: " << endl << p_vec.t() << endl;
    cout << "p: " << p << endl;
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "applying sub-sample transformation." << endl;
    // Adjust the GUS coefficients based on the Bernoulli sub-sample.
    for (uint32_t index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 * kNumRelations - CountBits(index));
    a *= pow(p, kNumRelations);
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    // Precomputing the coefficients.
    ComputeCoefficients(c_st, b);
    ComputeCoefficients(c_s, b);
    // This is done to account for the offset when computing the covariance. Now
    // the covariance is simply the dot product of the c and y coefficients.
    c_s(0) -= a * a;
    cout << "c_s: " << endl << c_s.t();
    cout << "c_s,t: " << endl << c_st;
  }

  // In both rounds, one worker per group is allocated.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    bool done = ++round == 1;
    cout << "Scheduling round: " << round << endl;
    for (uint32_t counter = 0; counter < num_groups; counter++)
      workers.push_back(WorkUnit(new LocalScheduler(counter), new cGLA(done)));
  }

  // In the first round
  void DoStep(Task& task, cGLA& gla) {
    if (round == 1) {
        SamplingGLA::Refilter(samples[task], *sub_samples[task], p);
    } else {
      // The various sums need for the final coefficients are computed.
      cout << "Task " << task << " has " << samples[task].size() << " elements." << endl;
      for (uint32_t mask = 0; mask < kNumCoefs; mask++) {
<?  if ($foreignKey) { ?>
        if (mask >= 2)
          continue;
<?  } ?>
        for (auto item : samples[task])
          Update(sums[task][mask], ChainHash(item.first, mask), item.second);
      }
      cout << "finished task: " << task << endl;
    }
  }

  // One fragment per distinct pair is allocated, including pairs of the same
  // group because their covariance is then the variance and not trivial.
  int GetNumFragments() {
    cout << "TIME TAKEN FOR COEFFICIENTS: " << timer.toc() << endl;
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
    cout <<"Starting entry: " << get<0>(*it) << " " << get<1>(*it) << endl;
    arma::vec::fixed<kNumCoefs> y(arma::fill::zeros);
    // Computing the pairwise product.
    for (uint32_t mask = 0; mask < kNumCoefs; mask++) {
<?  if ($foreignKey) { ?>
      if (mask >= 2)
        continue;
<?  } ?>
      auto& map_1 = sums[get<0>(*it)][mask];
      auto& map_2 = sums[get<1>(*it)][mask];
      for (auto item_1 : map_1) {
        auto item_2 = map_2.find(item_1.first);
        if (item_2 != map_2.end())
          y(mask) += item_1.second * item_2->second;
      }
    }
<?  if ($foreignKey) { ?>
    y.subvec(2, y.n_elem - 1).fill(y(1));
<?  } ?>
    cout << "  biased: " << y.t();
    UnbiasCoefficients(y, c_st);
    cout << "unbiased: " << y.t();
    // Returning the values.
    f = keys[std::get<0>(*it)];
    g = keys[std::get<1>(*it)];
    cov = dot(y, c_s);
    std::get<2>(*it) = true;
    cout << "Returned entry for matrix." << get<0>(*it) << " " << get<1>(*it) << endl;
    cout << "TIME TAKEN FOR RESULTS: " << timer.toc() << endl;
    return true;
  }

  void Finalize() {
    multi_it = std::make_tuple(0, 0, false);
  }

  bool GetNextResult(<?=typed_ref_args($outputs_)?>) {
    if (std::get<0>(multi_it) == num_groups)
      return false;
    GetNextResult(&multi_it, <?=args($outputs_)?>);
    std::get<2>(multi_it) = false;
    if (std::get<0>(multi_it) == std::get<1>(multi_it)) {
      std::get<0>(multi_it)++;
      std::get<1>(multi_it) = 0;
    } else {
      std::get<1>(multi_it)++;
    }
    return true;
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
