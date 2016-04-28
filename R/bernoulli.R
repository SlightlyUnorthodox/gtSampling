Bernoulli <- function(data, p, inputs, seed = NA, rng = "mt19937_64") {
  if (missing(inputs)) {
    gf <- GF(BernoulliSample, p = p, rng = rng)
    Filter(data, gf)
  } else {
    inputs <- convert.exprs(substitute(inputs))
    if (length(inputs) != 1)
      stop("A deterministic Bernoulli sample should have exactly 1 input.")
    gf <- GF(sampling::Deterministic_Bernoulli, p = p, seed = seed)
    Filter(data, gf, inputs)
  }
}
