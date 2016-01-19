<?
// Resources:
// armadillo: vectors
// math.h: pow
// unordered_map: unordered_map
function Covariance($t_args, $outputs, $states)
{
    // Class name is randomly generated.
    $className = generate_name('Covariance');

    // Processing of template arguments.
    $a = $t_args['a'];
    $b = $t_args['b'];

    $numCoefs     = count($b);
    $numRelations = log($numCoefs, 2);
    $bElems       = implode(', ', $b);

    // Processing of input states.
    $states_  = array_combine(['state_x', 'state_y'], $states);
    $sample_x = array_get_index($states_['state_x']->get('glas'), 1);
    $sample_y = array_get_index($states_['state_y']->get('glas'), 1);

    // Processing of outputs.
    $outputs = array_fill_keys(array_keys($outputs), lookupType('double'));
    if (count($outputs) == 1) {
        $outputs_ = array_combine(['covariance'], $outputs);
        $outputCoefs = false;
    } else {
        $outputs_ = array_combine(['s', 'y_s', 'c_s', 'a', 'b_s'], $outputs);
        $outputCoefs = true;
    }

    $sys_headers  = ['armadillo', 'math.h', 'unordered_map'];
    $user_headers = [];
    $lib_headers  = [];
    $libraries    = [];
    $extra        = [];
    $result_type  = [$outputCoefs ? 'multi' : 'single'];
?>

using namespace std;
using namespace arma;

class AnswerGLA {
 private:
  // Whether this GLA should iterate.
  bool answer;

 public:
  AnswerGLA(bool answer) : answer(answer) {}
  void AddState(AnswerGLA other) {}
  bool ShouldIterate() { return answer; }
};

class <?=$className?>;

class <?=$className?> {
 public:
  using Mask = uint32_t;
  using Task = Mask;

  struct LocalScheduler {
    // The thread index of this scheduler.
    int index;

    // Whether this scheduler has scheduled its single task.
    bool finished;

    LocalScheduler(int index)
        : index(index),
          finished(false) {
    }

    bool GetNextTask(Task& task) {
      bool ret = !finished;
      // printf("Getting task from scheduler %d: %d\n", index, ret);
      task = index;
      finished = true;
      return ret;
    }
  };

  // The inner GLA being used.
  using cGLA = AnswerGLA;

  // The type of the workers.
  using WorkUnit = pair<LocalScheduler*, cGLA*>;

  // The type of the container for the workers.
  using WorkUnits = std::vector<WorkUnit>;

  // The type of GLA used for the sub-sampling.
  using XSamplingGLA = <?=$sample_x?>;
  using YSamplingGLA = <?=$sample_y?>;
  using SampleX = XSamplingGLA::Sample;
  using SampleY = YSamplingGLA::Sample;

  // The type on information being aggregated.
  using Value = double;

  // The type of the key hashing.
  using HashType = uint64_t;

  // The type of mapping used to compute each coefficient.
  using Map = std::unordered_map<HashType, Value>;

  // The number of relations.
  static const constexpr int kNumRelations = <?=$numRelations?>;

  // The number of coefficients, 2 ^ kNumRelations.
  static const constexpr int kNumCoefs = <?=$numCoefs?>;

  // The type used for the various arrays.
  using Array = arma::vec::fixed<kNumCoefs>;

 private:
  // The sub-samples used as inputs.
  SampleX sample_x;
  SampleY sample_y;

  // The GUS probability coefficient.
  double a;

  // The GUS, sample, and unbiased coefficients.
  Array b, y, y_hat;

  // A hash map containing the c_s,t coefficients.
  Map c;

<?  if ($outputCoefs) { ?>
  // The index used for the multi-type result.
  int index;
<?  } ?>

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : a(<?=$a?>),
        b({<?=$bElems?>}) {
    // Combining the two samples.
    auto x = state_x.GetGLA1();
    auto y = state_y.GetGLA1();
    auto p_x = x.GetProbability();
    auto p_y = y.GetProbability();
    if (p_x < p_y) {
      sample_x = x.GetSample();
      YSamplingGLA::Refilter(sample_y, y.GetSample(), p_x);
    } else if (p_x > p_y) {
      XSamplingGLA::Refilter(sample_x, x.GetSample(), p_y);
      sample_y = y.GetSample();
    } else {
      sample_x = x.GetSample();
      sample_y = y.GetSample();
    }
    // Adjust the GUS coefficients based on the Bernoulli sub-sample.
    double p = min(p_x, p_y);
    for (Mask index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 - (double) CountBits(index) / kNumRelations);
    a *= p;
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "sample_x size: " << sample_x.size() << endl;
  }

  // One worker per power set element S is allocate. This means that there are
  // 2^n workers where n is the number of relations and each worker computes a
  // single coefficient, Y_S. The value of S is encoded as a bit mask and stored
  // as an integer.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    for (Mask counter = 0; counter < kNumCoefs; counter++)
      workers.push_back(WorkUnit(new LocalScheduler(counter), new cGLA(false)));
  }

  // A groub-by is performed for each sample. The y coefficient is the sum of
  // the pairwise products across the two groupings.
  void DoStep(Task& task, cGLA& gla) {
    Map map_x(sample_x.size()), map_y(sample_y.size());

    for (auto item : sample_x)
      Update(map_x, ChainHash(item.first, task), item.second);

    for (auto item : sample_y)
      Update(map_y, ChainHash(item.first, task), item.second);
    cout << "Task: " << task << " Num Groups x: " << map_x.size() << " Num Groups y: " << map_y.size() << endl;

    for (auto it_x : map_x) {
      auto it_y = map_y.find(it_x.first);
      if (it_y != map_y.end())
        y[task] += it_x.second * it_y->second;
    }
  }

<?  if ($outputCoefs) { ?>
  void Finalize() {
    index = 0;
    ComputeCoefficients();
  }

  bool GetNextResult(<?=typed_ref_args($outputs_)?>) {
    if (index == kNumCoefs)
      return false;
    s = index;
    y_s = y_hat[s];
    c_s = ComputeCCoefficient(s);
    a = a;
    b_s = b[s];
    index++;
    return true;
  }
<?  } else { ?>
  void GetResult(<?=typed_ref_args($outputs_)?>) {
    ComputeCoefficients();
    covariance = 0;
    for (Mask s = 0; s < kNumCoefs; s++)
      covariance += ComputeCCoefficient(s) * y_hat[s];
    covariance -= pow(a, 2) * y_hat[0];
    cout << "covariance: " << covariance << endl;\
  }
<?  } ?>

 private:
  // This updates a map given a value and key.
  void Update(Map& map, HashType key, Value value) {
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
  HashType ChainHash(XSamplingGLA::KeySet keys, Mask mask) {
    HashType result = mask;
<?  for ($index = 0; $index < $numRelations; $index++) { ?>
    if (mask & 1 << <?=$index?>)
        result = CongruentHash(result, Hash(get<<?=$index?>>(keys)));
<?  } ?>
    //cout << "Keys: ";
<?  for ($index = 0; $index < $numRelations; $index++) { ?>
    //cout << get<<?=$index?>>(keys) << " ";
<?  } ?>
    //cout << endl << "Mask: " << mask << " Result: " << result << endl;
    return result;
  }

  // This function computes the values of y_hat based on y. Because the mask is
  // decreasing during the loop, it is guaranteed that y_hat[s | t] has already
  // been computed, as s | t >= s regardless of s and t.
  void ComputeCoefficients() {
    cout << "Biased coefficients: "<< y.t();
    // Because s is unsigned, it will roll over once decremented past 0.
    for (Mask s = kNumCoefs - 1; s < kNumCoefs; s--) {
      double sum = 0;
      for (Mask t = 1; t <= kNumCoefs - 1; t++) {
        if (t & s)  // t is not a subset of the complement of s.
          continue;
        sum += ComputeCCoefficient(s, t) * y_hat[s | t];
      }
      y_hat[s] = (y[s] - sum) / ComputeCCoefficient(s, 0);
    }
    cout << "Unbiased coefficients: " << y_hat.t();
  }

  // This function computes the value of c_s,t given s and t as bit masks. The
  // value is then cache to avoid repeated computation.
  double ComputeCCoefficient(Mask s, Mask t) {
    // First check if the value has been computed.
    HashType key = ((HashType) s << 32) + t;
    auto it = c.find(key);
    if (it != c.end()) {
      cout << "Retrieved hashed value for s = " << s << " and t = " << t << endl;
      cout << "Value is " << it->second;
      return it->second;
    }
    // The value has not been computed.
    double coef = 0;
    cout << "computing coef: " << s << " " << t << endl;
    for (Mask u = 0; u <= t; u++) {  // A bit mask representing the set u.
      if (u & ~t)  // u is not a subset of t and we ignore this case.
        continue;
      cout << "u = " << u << endl;
      // t is a subset of the complement of s, meaning they share no elements.
      // u is a subset of T, meaning it and s share no elements. Hence, |u U s|
      // is equivalent to |u| + |s|.
      cout << "bits = " << (CountBits(u | s) % 2) << endl;
      cout << "mult = " << (1 - 2 * (CountBits(u | s) % 2)) << endl;
      cout << "b_su = " << b[s | u] << endl;
      // This differs from the vldb paper which says the exponent is |s| + |u|.
      // This is intentional as that is a typo. It should read |t \ u| which is
      // equivalent to |t - u|, as u is a subset of t.
      coef += (1 - 2 * (CountBits(t - u) % 2)) * b[s | u];
      cout << "coef = " << coef << endl;
    }
    cout << "result: " << coef << endl << endl;;
    c.insert(std::make_pair(key, coef));
    return coef;
  }

  // This function computes the value of c_s given s as a bit mask. Unlike the
  // previous function, no values are cached as each c_s is only needed once.
  double ComputeCCoefficient(Mask s) {
    cout << "computing c_s for s = " << s << endl;
    double coef = 0;
    for (Mask t = 0; t <= s; t++) {
      if (t & ~s)  // t is not a subset of s.
        continue;
      // Because t is a subset of s, |t| + |s| = |s \ t| mod 2 because whichever
      // element appears in t will appear in s as well. This means that such an
      // element will be counted twice on the left hand side and therefore not
      // affect the parity of the result.
      cout << "t = " << t << endl;
      cout << "bits = " << (CountBits(s - t) % 2) << endl;
      cout << "mult = " << (1 - 2 * (CountBits(s - t) % 2)) << endl;
      cout << "b_t = " << b[t] << endl;
      cout << "change = " << (1 - 2 * (CountBits(s - t) % 2)) * b[t] << endl;
      cout << "coef = " << coef << endl;
      coef += (1 - 2 * (CountBits(s - t) % 2)) * b[t];
      cout << "coef = " << coef << endl;
    }
    cout << "c_s: " << coef << " s: " << s << endl << endl;;
    return coef;
  }

  // The bitwise-SWAR algorithm for computing the Hamming Weight of an integer.
  int CountBits(Mask i) {
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
