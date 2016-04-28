SampleWOR <- function(..., cardinality) {
  aggregate <- ReservoirSample(...)
  aggregate$gla$args$cardinality <- cardinality
  aggregate
}

## This marks a waypoint as having GUS coefficients which it normally wouldn't
## have. This is used for interpreting a table as a simulated sample of a larger
## database. Therefore, the given waypoint should only ever be a GI.
MarkSample <- function(waypoint, gus) {
  waypoint$gus <- gus
  waypoint
}
