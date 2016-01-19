<?
// This GF performs a Bernoulli sampling in which inclusion of an input tuple is
// determined entirely by its key and the seed, not by the ordering of chunks.

// Template Args:
// seed: The initial value used in the hashing, an integer between 0 and 10^9.

// Resources:
// limits: numeric_limits
// HashFct.h: CongruentHash

function Deterministic_Bernoulli($t_args, $inputs) {
    // Class name is randomly generated.
    $className = generate_name('DeterministicBernoulli');

    // Processing of inputs.
    $inputs_ = array_combine(['key'], $inputs);
    $key = $inputs_['key'];

    // Initialization of local variables from template arguments.
    $seed = $t_args['seed'];
    $prob = $t_args['p'];

    $sys_headers  = ['limits'];
    $user_headers = [];
    $lib_headers  = ['base\HashFct.h'];
    $libraries    = [];
    $extra        = [];
    $result_type  = ['fragment', 'multi'];
?>

using namespace std;

class <?=$className?>;

class <?=$className?> {
 public:
  // The type of the key hashing.
  using HashType = uint64_t;

  // The constand seed used to synchronize repeated samples.
  static const constexpr HashType kSeed = <?=$seed?>;

  // The probability a random key is passed through.
  static const constexpr long double kProbability = <?=$prob?>;

  // The maximum value of HashType.
  static const constexpr long double kMax = numeric_limits<HashType>::max();

  // The limit of the hashed value for which the input is kept.
  static const constexpr HashType kLimit = (kMax + 1) * kProbability - 1;

 public:
  <?=$className?>() {}

  bool Filter(<?=const_typed_ref_args($inputs_)?>) {
    return CongruentHash(Hash(key), kSeed) <= kLimit;
  }
};

<?
    return [
        'kind'           => 'GF',
        'name'           => $className,
        'system_headers' => $sys_headers,
        'user_headers'   => $user_headers,
        'lib_headers'    => $lib_headers,
        'libraries'      => $libraries,
        'extra'          => $extra,
        'iterable'       => false,
        'input'          => $inputs,
        'result_type'    => $result_type,
    ];
}
?>
