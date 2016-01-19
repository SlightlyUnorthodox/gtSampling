#' Computes a Covariance Matrix between a Collection of Filters
#'
#' The exterior query is split into sub-queries based on its grouping
#' expressions. Each of these sub-queries is then joined with the interior query
#' and the covariance is computed pair-wise across the various groups.
#'
#' @param interior A \code{\link{waypoint}} that represents the shared portion
#'   of the query.
#' @param exterior A \code{\link{waypoint}} that represents the non-shared
#'   portions of the query.
#' @param group A named list of expressions, with the names being used as the
#'   corresponding outputs. These expressions are outputted in addition the
#'   results of the inner GLAs.
#'
#'   If no name is given and the corresponding  expression is simply an
#'   attribute, then said attribute is used as the name. Otherwise, the column
#'   for that expression is hidden from the user.
#' @param keys A list of expressions with exactly one expression per relation in
#'   the overall query. Each expression should represent a key for the
#'   corresponding relation.
CovarianceMatrix <- function(interior, exterior, inAtts, exAtts,
                             group, keys, value, outputs) {
  ## The join is performed first.
  data <- do.call("Join", list(substitute(interior), substitute(inAtts),
                               substitute(exterior), substitute(exAtts)))

  ## The necessary GUS coefficients are computed for each branch.
  b <- CompressGUS(interior)$b
  a <- CompressGUS(exterior)$a
  ## The GUS coefficients are combined using rule 2. The exterior parts have the
  ## same GUS coefficients because they only differ by a filter.
  gus <- lapply(list(a = b[[1]], b = b), `*`, a ^ 2)

  keys <- substitute(keys)
  value <- substitute(value)
  group <- substitute(group)
  l <- length(convert.exprs(keys, data))

  ## Constructing the function calls manually. Dummy names are used to avoid
  ## name clashing and to ensure complex inputs are given output names.
  mean <- call("Sum", value, substitute(junk1))
  sample <- call("AdjBernoulli", keys, c(junk2 = value))

  if (l != log(length(gus$b), 2))
    stop("incorrect number of keys given: ", l)

  outputs <- convert.atts(substitute(outputs))

  data <- do.call("GroupBy", list(quote(data), group, sample))
  GIST <- GIST(sampling::Covariance_Matrix, a = gus$a, b = gus$b)
  Transition(GIST, outputs, data)
}

CompressGUS <- function(waypoint) UseMethod("CompressGUS")

CompressGUS.default <- function(waypoint)
  stop("unsupported waypoint used in sampling query. class: ",
       paste(class(waypoint), collapse = ", "))

CompressGUS.GI <- CompressGUS.Load <- function(waypoint) list(a = 1, b = c(1, 1))

CompressGUS.Generated <- CompressGUS.Cache <- CompressGUS.Filter <- function(waypoint)
  CompressGUS(waypoint$data)

CompressGUS.GF <- function(waypoint) {
  if ((name <- get.function.name(waypoint$gf)) != "base::BernoulliSample")
    stop("illegal GF used in sampling query: ", name)
  p <- waypoint$gf$args$p
  gus <- list(a = p, b = c(p * p, p))
  mapply(`*`, gus, CompressGUS(waypoint$data))
}

CompressGUS.GLA <- function(waypoint) {
  if ((name <- get.function.name(waypoint$gla)) != "statistics::Reservoir_Sampling")
    stop("illegal GLA used in sampling query: ", name)
  size <- waypoint$gla$args$size
  card <- waypoint$gla$args$cardinality
  assert(!is.null(card),
         "Cardinality not given for reservoir sampling.")
  gus <- list(a = size / card, b = c(size * (size - 1) / card / (card - 1), size / card))
  mapply(`*`, gus, CompressGUS(waypoint$data))
}

CompressGUS.Join <- function(waypoint) {
  x <- CompressGUS(waypoint$x)
  y <- CompressGUS(waypoint$y)
  list(a = x$a * y$a, b = as.numeric(matrix(y$b) %*% t(matrix(x$b))))
}
