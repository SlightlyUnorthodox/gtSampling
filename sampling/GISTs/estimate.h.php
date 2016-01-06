<?
// This currently only performs estimation of a single summation aggregate.

// For the various computational details, see:
//    http://www.vldb.org/pvldb/vol6/p1798-nirkhiwale.pdf

// The input to this GIST is a multiplexer state that performed both aggregation
// and a Bernoulli sub-sampling. The input to the multiplexer is the sample that
// is being used to estimate the population's characteristics. The Bernoulli sub
// sample is itself a GUS and is used to reduce the size of the sample needed to
// compute the Y_s terms. Because it is a GUS, we can compact it and its input
// into a single GUS. The advantage of sub-sampling is that the variance of the
// estimators is reduced by having a large sample and the computation performed
// to compute the coefficients is reduced by having a small sub-sample.

// The first step done is to compact the Bernoulli GUS on top of its input. This
// is done like any other GUS compact. However, the Bernoulli sub-sample is not
// a simple sample but actually represents the joining of independent Bernoulli
// samples on each of the original relations. Let a be the parameter to the sub
// sample and n the number of relations. Then each independent sample on top of
// a single relation is a Bernoulli sample with parameter a' = a ^ 1/n.

// The GUS coefficient b for one of those n sub-samples is b' = [a' ^ 2, a'] ^ n
// Hence, b_P(n) = f( ... f(f(b', b'), b') ... ), where f is the operation used
// to combine 2 b parameters across a join and is repeated n - 1 times. This can
// be simplied to b_s = a' ^ (2 * n - |s|) = a ^ (2 - |s| / n) for s in P(n).

// The second step is to compute each y_s term, performed during the first and
// only iteration with each worker computing a single term. Each grouping uses
// its own hash table, with no cross lookups currently being done.

// The final step is to unbias the coefficients. This computation is relatively
// simple and is done on a single thread immediately before the output.

// The output is either the various coefficients (s, y_s, c_s, a, b_s) or the
// expected value and its standard deviation.

// Resources:
// armadillo: vectors
// math.h: pow
// unordered_map: unordered_map
function Estimate($t_args, $outputs, $states)
{
    // Class name is randomly generated.
    $className = generate_name('Estimate');

    // Initialization of local variables from template arguments.
    $a = $t_args['a'];
    $b = $t_args['b'];

    grokit_assert(is_numeric($a), 'Parameter `a` must be numeric.');
    grokit_assert(is_array($b), 'Parameter `b` must be an array.');
    grokit_assert(!in_array(false, array_map('is_numeric', $b)),
                  'Parameter `b` should only contain number.');

    $numCoefs     = count($b);
    $numRelations = log($numCoefs, 2);
    $bElems       = implode(', ', $b);

    $states_ = array_combine(['input'], $states);
    $input  = $states_['input'];
    $glas   = $input->get('glas');
    $mean   = array_get_index($glas, 0);
    $sample = array_get_index($glas, 1);
    $inputs = $sample->input();
    $keys   = array_slice($inputs, 0, -1);

    // Processing of outputs.
    $outputs = array_fill_keys(array_keys($outputs), lookupType('double'));
    if (count($outputs) == 2) {
        $outputs_ = array_combine(['exp_val', 'variance'], $outputs);
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
  using WorkUnits = vector<WorkUnit>;

  // The type of GLA used for the sub-sampling.
  using SamplingGLA = <?=$sample?>;

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
  // The GLA that performed the sub-sampling.
  SamplingGLA sample;

  // The size of the sub-sample.
  long size;

  // The GUS probability coefficient.
  double a;

  // The GUS, sample, and unbiased coefficients.
  Array b, y, y_hat;

  // The sample's sum of the property whose sum is being estimated.
  Value sum;

  // A hash map containing the c_s,t coefficients.
  Map c;

<?  if ($outputCoefs) { ?>
  // The index used for the multi-type result.
  int index;
<?  } ?>

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : sample(input.GetGLA1()),
        size(sample.GetSize()),
        a(<?=$a?>),
        b({<?=$bElems?>}) {
    input.GetGLA0().GetResult(sum);
    // Adjust the GUS coefficients based on the Bernoulli sub-sample.
    double p = sample.GetProbability();
    for (Mask index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 - (double) CountBits(index) / kNumRelations);
    a *= p;
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "sum: " << sum << endl;
    cout << "size: " << size << endl;
    cout << "sample: " << endl;
    for (auto item : sample.GetSample())
      cout << get<0>(item.first) << " " << get<1>(item.first) << " " << item.second << endl;
  }

  // One worker per power set element S is allocate. This means that there are
  // 2^n workers where n is the number of relations and each worker computes a
  // single coefficient, Y_S. The value of S is encoded as a bit mask and stored
  // as an integer.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    for (Mask counter = 0; counter < kNumCoefs; counter++)
      workers.push_back(WorkUnit(new LocalScheduler(counter), new cGLA(false)));
  }

  void DoStep(Task& task, cGLA& gla) {
    Map map(size);
    for (auto item : sample.GetSample()) {
      HashType key = ChainHash(item.first, task);
      Map::iterator it = map.find(key);
      if (it == map.end()) {
        auto insertion = map.insert(Map::value_type(key, 0));
        it = insertion.first;
      }
      it->second += item.second;
    }
    cout << "Task: " << task << " Num Groups: " << map.size() << endl;
    for (auto group : map)
      y[task] += pow(group.second, 2);
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
    exp_val = sum / a;
    variance = 0;
    for (Mask s = 0; s < kNumCoefs; s++)
      variance += ComputeCCoefficient(s) * y_hat[s];
    variance /= pow(a, 2);
    variance -= y_hat[0];
    cout << "variance: " << variance << endl;
    cout << "exp_val: " << exp_val << endl;
  }
<?  } ?>

 private:
  // This function takes in a set of keys and a bit mask specifying which keys
  // are to be used. It hashes each relevent key to a uint64_t and then uses a
  // chain hash to condense these hashed values into a single output. The mask
  // is used as the starting value for the chain hash.
  HashType ChainHash(SamplingGLA::KeySet keys, Mask mask) {
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
      cout << "mult = " << (1 - 2 * (int) (CountBits(u | s) % 2)) << endl;
      cout << "b_su = " << b[s | u] << endl;
      // This differs from the vldb paper which says the exponent is |s| + |u|.
      // This is intentional as that is a typo. It should read |t \ u| which is
      // equivalent to |t - u|, as u is a subset of t.
      coef += (1 - 2 * (int) (CountBits(t - u) % 2)) * b[s | u];
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
      cout << "mult = " << (1 - 2 * (int) (CountBits(s - t) % 2)) << endl;
      cout << "b_t = " << b[t] << endl;
      cout << "change = " << (1 - 2 * (int) (CountBits(s - t) % 2)) * b[t] << endl;
      cout << "coef = " << coef << endl;
      coef += (1 - 2 * (int) (CountBits(s - t) % 2)) * b[t];
      cout << "coef = " << coef << endl;
    }
    cout << "c_s: " << coef << " s: " << s << endl << endl;;
    return coef;
  }

  // The bitwise-SWAR algorithm for computing the Hamming Weight of an integer.
  Mask CountBits(Mask i) {
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
