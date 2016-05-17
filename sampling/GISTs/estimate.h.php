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
//   armadillo: vectors
//   math.h: pow
//   unordered_map: unordered_map
function Estimate($t_args, $outputs, $states)
{
    // Class name is randomly generated.
    $className = generate_name('Estimate');

    // Processing of template arguments.
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
    $lib_headers  = ['coefs.h', 'tools.h', 'base\gist.h'];
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

  // The GLA and container used for the sub-sampling.
  using SamplingGLA = <?=$sample?>;
  using Sample = SamplingGLA::Sample;
  using Value = SamplingGLA::Value;

  // The type of mapping used to compute each coefficient.
  using Map = std::unordered_map<uint64_t, Value>;

  // The number of relations.
  static const constexpr int kNumRelations = <?=$numRelations?>;

  // The number of coefficients, 2 ^ kNumRelations.
  static const constexpr int kNumCoefs = <?=$numCoefs?>;

 private:
  // The sample's sum of the property whose total sum is being estimated.
  Value sum;

  // The GLA that performed the sub-sampling.
  Sample sample;

  // The GUS probability coefficient.
  double a;

  // Containers for the various coefficients
  arma::vec::fixed<kNumCoefs> b, y;
  arma::mat::fixed<kNumCoefs, kNumCoefs> c_st;
  arma::vec::fixed<kNumCoefs> c_s;

<?  if ($outputCoefs) { ?>
  // The index used for the multi-type result.
  int index;
<?  } ?>

 public:
  <?=$className?>(<?=const_typed_ref_args($states_)?>)
      : sample(input.GetGLA1().GetSample()),
        a(<?=$a?>),
        b({<?=$bElems?>}) {
    // Copyiny the value of the total sum.
    input.GetGLA0().GetResult(sum);
    sum /= a;
    // Precomputing the coefficients.
    ComputeCoefficients(c_st, b);
    ComputeCoefficients(c_s, b);
    // Adjust the GUS coefficients based on the Bernoulli sub-sample.
    double p = input.GetGLA1().GetProbability();
    for (uint32_t index = 0; index < kNumCoefs; index++)
      b[index] *= pow(p, 2 * kNumRelations - CountBits(index));
    a *= pow(p, kNumRelations);
    cout << "a: " << a << endl;
    cout << "b: " << b.t();
    cout << "sum: " << sum << endl;
    cout << "sample: " << endl;
    for (auto item : sample)
      cout << get<0>(item.first) << " " << get<1>(item.first) << " " << item.second << endl;
  }

  // One worker per power set element S is allocate. This means that there are
  // 2^n workers where n is the number of relations and each worker computes a
  // single coefficient, Y_S. The value of S is encoded as a bit mask and stored
  // as an integer.
  void PrepareRound(WorkUnits& workers, int num_threads) {
    for (uint32_t counter = 0; counter < kNumCoefs; counter++)
      workers.push_back(WorkUnit(new LocalScheduler(counter), new cGLA()));
  }

  void DoStep(Task& task, cGLA& gla) {
    Map map(sample.size());

    for (auto item : sample)
      Update(map, ChainHash(item.first, task), item.second);

    cout << "Task: " << task << " Num Groups: " << map.size() << endl;
    for (auto group : map)
      y[task] += pow(group.second, 2);
  }

<?  if ($outputCoefs) { ?>
  void Finalize() {
    index = 0;
    UnbiasCoefficients(y, c_st);
  }

  bool GetNextResult(<?=typed_ref_args($outputs_)?>) {
    if (index == kNumCoefs)
      return false;
    s = index;
    y_s = y(s);
    c_s = c_s(s);
    a = a;
    b_s = b(s);
    index++;
    return true;
  }
<?  } else { ?>
  void GetResult(<?=typed_ref_args($outputs_)?>) {
    UnbiasCoefficients(y, c_st);
    exp_val = sum;
    variance = 0;
    for (uint32_t s = 0; s < kNumCoefs; s++)
      variance += c_s(s) * y(s);
    variance /= pow(a, 2);
    variance -= y(0);
    cout << "variance: " << variance << endl;
    cout << "exp_val: " << exp_val << endl;
  }
<?  } ?>
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
