SampleWOR <- function(..., cardinality) {
  aggregate <- ReservoirSample(...)
  aggregate$gla$args$cardinality <- cardinality
  aggregate
}
