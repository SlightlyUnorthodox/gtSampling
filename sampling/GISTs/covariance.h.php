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
    $lib_headers  = ['coefs.h', 'base\HashFct.h', 'base\gist.h'];
    $libraries    = [];
    $extra        = [];
    $result_type  = [$outputCoefs ? 'multi' : 'single'];
?>

using namespace std;
using namespace arma;

class <?=$className?>;

class <?=$className?> {
 public:
  // The various components for the GIST and Fragment result type.
  using cGLA = HaltingGLA;
  using Task = uint32_t;
  using LocalScheduler = SimpleScheduler<Task>;
  using WorkUnit = std::pair<LocalScheduler*, cGLA*>;
  using WorkUnits = std::vector<WorkUnit>;
  using Iterator = std::tuple<int, int, bool>;

  // The type of GLA used for the sub-sampling.
  using XSamplingGLA = <?=$sample_x?>;
  using YSamplingGLA = <?=$sample_y?>;
  using SampleX = XSamplingGLA::Sample;
  using SampleY = YSamplingGLA::Sample;

  // The type on information being aggregated.
  using Value = long double;

  // The type of mapping used to compute each coefficient.
  using Map = std::unordered_map<uint64_t, Value>;

  // The number of relations.
  static const constexpr int kNumRelations = <?=$numRelations?>;

  // The number of coefficients, 2 ^ kNumRelations.
  static const constexpr int kNumCoefs = <?=$numCoefs?>;

 private:
  // The sub-samples used as inputs.
  SampleX sample_x;
  SampleY sample_y;

  // The GUS probability coefficient.
  double a;

  // The GUS, sample, and unbiased coefficients.
  arma::vec::fixed<kNumCoefs> b, y;

  // Containers for the precomputed c coefficients.
  arma::vec::fixed<kNumCoefs> c_s;
  arma::mat::fixed<kNumCoefs, kNumCoefs> c_st;

<?  if ($outputCoefs) { ?>
  // The index used for the multi-type result.
  int index;
<?  } ?>

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : a(<?=$a?>),
        b({<?=$bElems?>}) {
    // Precomputing the coefficients.
    ComputeCoefficients(c_st, b);
    ComputeCoefficients(c_s, b);
    cout << c_st << endl << c_s.t();
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
    cout << "p: " << p << endl;
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "applying sub-sample transformation." << endl;
    for (uint32_t index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 - (double) CountBits(index) / kNumRelations);
    a *= p;
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "sample_x size: " << sample_x.size() << endl;
    long sum1; double sum2;
    state_x.GetGLA0().GetResult(sum1);
    state_y.GetGLA0().GetResult(sum2);
    cout << "sum_x: " << sum1 << endl << "sum_y: " << sum2 << endl;
  }

  // One worker per power set element S is allocate. This means that there are
  // 2^n workers where n is the number of relations and each worker computes a
  // single coefficient, Y_S. The value of S is encoded as a bit mask and stored
  // as an integer.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    for (uint32_t counter = 0; counter < kNumCoefs; counter++)
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
    UnbiasCoefficients(y, c_st, b);
  }

  bool GetNextResult(<?=typed_ref_args($outputs_)?>) {
    if (index == kNumCoefs)
      return false;
    s = index;
    y_s = y_hat[s];
    c_s = c_s(s);
    a = a;
    b_s = b[s];
    index++;
    return true;
  }
<?  } else { ?>
  void GetResult(<?=typed_ref_args($outputs_)?>) {
    cout << "biased: " << y.t();
    UnbiasCoefficients(y, c_st, b);
    cout << "unbiased: " << y.t();
    covariance = 0;
    for (uint32_t s = 0; s < kNumCoefs; s++)
      covariance += c_s(s) * y(s);
    covariance -= pow(a, 2) * y(0);
    cout << "covariance: " << covariance << endl;
  }
<?  } ?>

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
  uint64_t ChainHash(XSamplingGLA::KeySet keys, uint32_t mask) {
    uint64_t result = mask;
<?  for ($index = 0; $index < $numRelations; $index++) { ?>
    if (mask & 1 << <?=$index?>)
        result = CongruentHash(result, Hash(get<<?=$index?>>(keys)));
<?  } ?>
    //cout << "Keys: ";
<?  for ($index = 0; $index < $numRelations; $index++) { ?>
    //cout << get<<?=$index?>>(keys) << " ";
<?  } ?>
    //cout << endl << "uint32_t: " << mask << " Result: " << result << endl;
    return result;
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
