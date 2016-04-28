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

    $foreignKey = $t_args['foreign.key'];

    // Processing of input states.
    $states_  = array_combine(['state_x', 'state_y'], $states);
    $sample_x = array_get_index($states_['state_x']->get('glas'), 1);
    $sample_y = array_get_index($states_['state_y']->get('glas'), 1);
    $type_x = array_get_index(array_get_index($states_['state_x']->get('glas'), 0)->output(), 0);
    $type_y = array_get_index(array_get_index($states_['state_y']->get('glas'), 0)->output(), 0);

    // Processing of outputs.
    $outputs = array_fill_keys(array_keys($outputs), lookupType('double'));
    $count = count($outputs);
    $outputVariance = $count >= 3;
    $outputMean = $count >= 5;
    $outputStat = $count >= 8;
    switch($count) {
    case 1:
        $names = ['covariance'];
        break;
    case 3:
        $names = ['covariance', 'var_x', 'var_y'];
        break;
    case 5:
        $names = ['covariance', 'var_x', 'var_y', 'mean_x', 'mean_y'];
        break;
    case 8:
        $names = ['covariance', 'var_x', 'var_y', 'mean_x', 'mean_y', 'time', 'x_count', 'y_count'];
        break;
    default:
        grokit_error("$count inputs given to sampling\covariance GIST");
        break;
    }
    $outputs_ = array_combine($names, $outputs);

    $sys_headers  = ['armadillo', 'math.h', 'unordered_map'];
    $user_headers = [];
    $lib_headers  = ['coefs.h', 'tools.h', 'base\gist.h'];
    $libraries    = [];
    $extra        = [];
    $result_type  = ['single'];
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

  // The GLA and container used for the sub-sampling.
  using XSamplingGLA = <?=$sample_x?>;
  using YSamplingGLA = <?=$sample_y?>;
  using SampleX = XSamplingGLA::Sample;
  using SampleY = YSamplingGLA::Sample;

  // The type of mapping used to compute each coefficient.
  using Map = std::unordered_map<uint64_t, long double>;

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

  // Containers for the various coefficients
  arma::vec::fixed<kNumCoefs> b, y;
  arma::vec::fixed<kNumCoefs> c_s;
  arma::mat::fixed<kNumCoefs, kNumCoefs> c_st;

<?  if ($outputVariance) { ?>
  // Containers for the y_s terms for the variances.
  arma::vec::fixed<kNumCoefs> y_x, y_y;

<?      if ($outputMean) { ?>
  // The sum of the sampled values.
  <?=$type_x?> sum_x;
  <?=$type_y?> sum_y;
<?      } ?>
<?  } ?>

<?  if ($outputStat) { ?>
  long count_x, count_y;
<?  } ?>

  arma::wall_clock timer;

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : a(<?=$a?>),
        b({<?=$bElems?>}) {
    timer.tic();
    // Combining the two samples.
    auto x = state_x.GetGLA1();
    auto y = state_y.GetGLA1();
    auto p_x = x.GetProbability();
    auto p_y = y.GetProbability();
    cout << "p_x: " << p_x << " p_y: " << p_y << endl;
    cout << "sample_x size: " << x.GetSample().size() << endl;
    cout << "sample_y size: " << y.GetSample().size() << endl;
    // for (auto item : x.GetSample())
    //   cout << "Keys: " << item.first << " Value: " << item.second << endl;
    // for (auto item : y.GetSample())
    //   cout << "Keys: " << item.first << " Value: " << item.second << endl;
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
<?  if ($outputMean) { ?>
    state_x.GetGLA0().GetResult(sum_x);
    state_y.GetGLA0().GetResult(sum_y);
    cout << "sum_x: " << sum_x << endl << "sum_y: " << sum_y << endl;
    sum_x /= a;
    sum_y /= a;
<?  } ?>
<?  if ($outputStat) { ?>
    count_x = x.GetCount();
    count_y = y.GetCount();
<?  } ?>
    // Adjust the GUS coefficients based on the Bernoulli sub-sample.
    double p = min(p_x, p_y);
    cout << "p: " << p << endl;
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "applying sub-sample transformation." << endl;
    for (uint32_t index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 * kNumRelations - CountBits(index));
    a *= pow(p, kNumRelations);
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "sample_x size: " << sample_x.size() << endl;
    cout << "sample_y size: " << sample_y.size() << endl;
    // b = arma::vec({2.5e-05,5.0e-05,2.5e-04,5.0e-04,2.5e-04,5.0e-04,2.5e-03,5.0e-03});
    // Precomputing the coefficients.
    ComputeCoefficients(c_st, b);
    ComputeCoefficients(c_s, b);
    // This is done to account for the offset when computing the covariance. Now
    // the covariance is simply the dot product of the c and y coefficients.
    c_s(0) -= a * a;
    cout << c_st << endl << c_s.t();
  }

  // One worker per power set element S is allocate. This means that there are
  // 2^n workers where n is the number of relations and each worker computes a
  // single coefficient, Y_S. The value of S is encoded as a bit mask and stored
  // as an integer.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    for (uint32_t counter = 0; counter < kNumCoefs; counter++)
      workers.push_back(WorkUnit(new LocalScheduler(counter), new cGLA()));
  }

  // A groub-by is performed for each sample. The y coefficient is the sum of
  // the pairwise products across the two groupings.
  void DoStep(Task& task, cGLA& gla) {
<?  if ($foreignKey) { ?>
    if (task >= 2)
      return;
<?  } ?>

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

<?  if ($foreignKey) { ?>
    if (task == 1)
      y.subvec(2, y.n_elem - 1).fill(y[task]);
<?  } ?>


<?  if ($outputVariance) { ?>
    for (auto group : map_x)
      y_x[task] += pow(group.second, 2);

    for (auto group : map_y)
      y_y[task] += pow(group.second, 2);
<?  } ?>
  }

  void GetResult(<?=typed_ref_args($outputs_)?>) {
    cout << "biased: " << y.t();
    UnbiasCoefficients(y, c_st);
    cout << "unbiased: " << y.t();
    covariance = dot(y, c_s);
<?  if ($outputVariance) { ?>
    // y_y = arma::vec({4.5158e+12, 1.6472e+11, 1.1366e+11, 1.1366e+11, 9.0588e+10, 9.0588e+10, 9.0588e+10, 9.0588e+10});
    cout << "biased left: " << y_x.t();
    cout << "biased right: " << y_y.t();
    UnbiasCoefficients(y_x, c_st);
    UnbiasCoefficients(y_y, c_st);
    cout << "unbiased left: " << y_x.t();
    cout << "unbiased right: " << y_y.t();
    var_x = dot(y_x, c_s) / pow(a, 2);
    var_y = dot(y_y, c_s) / pow(a, 2);
<?      if ($outputMean) { ?>
    mean_x = sum_x;
    mean_y = sum_y;
<?      } ?>
<?  } ?>
<?  if ($outputStat) { ?>
    time = timer.toc();
    x_count = count_x;
    y_count = count_y;
<?  } ?>
    cout << "covariance: " << covariance << endl;
    cout << "TIME TAKEN FOR COEFFICIENTS: " << timer.toc() << endl;
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
