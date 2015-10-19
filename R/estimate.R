Estimate <- function(data, keys, value, outputs) {
  keys <- substitute(keys)
  value <- substitute(value)

  ## Constructing the function calls manually.
  mean <- call("Sum", value, substitute(junk))
  sample <- call("AdjBernoulli", keys, value)
  multi <- do.call("Multiplexer", list(substitute(data), mean, sample))

  outputs <- convert.atts(substitute(outputs))

  GIST <- GIST(sampling::Estimate)
  Transition(GIST, outputs, multi)
}
