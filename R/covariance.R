## shared should be a character vector mapping aliases from state 1 to state 2.
Covariance <- function(states, keys, value1, value2, outputs, shared, min = 10^5, foreign.key = FALSE) {
  ## Examining the query tree. The inputs are labelled x and y.
  x <- states[[1]]
  y <- states[[2]]
  ## u, w, v are the exclusive parts of x, the exclusive parts of y, and the
  ## shared parts of them, respectively.
  if (length(intersect(Parents(x), Parents(y))) > 0)
    stop("the given states share some waypoints.")
  shared <- setNames(Shared(y, unname(shared)), Shared(x, names(shared)))
  v <- c(unname(shared), names(shared))
  u <- setdiff(Parents(x), v)
  w <- setdiff(Parents(y), v)

  ## Returns -1, 0, 1 for alias in u, v, w respectively.
  Match <- function(alias)
    which(sapply(list(u, v, w), is.element, el = alias)) - 2

  ## Stripping the important parts from each waypoint, which are the alias, the
  ## GUS coefficients, the operator, and the input waypoints.
  Clean <- function(waypoint) {
    ## Computing the operator of this waypoint.
    allowed <- c("Join", "GLA", "GF", "Filter", "GI", "Load", "Generated")
    classes <- class(waypoint)
    op <- classes[classes %in% allowed][[1]]
    alias <- waypoint$alias
    if (alias %in% names(shared))
      alias <- shared[[alias]]
    branch <- Match(alias)

    gus <-
      if (op == "GLA") {
        size = waypoint$gla$args$size
        card = waypoint$gla$args$cardinality
        list(a = size / card,
             b = c(size * (size - 1) / card / (card - 1), size / card))
      } else if (op == "GF") {
        p = waypoint$gf$args$p
        list(a = p, b = c(p * p, p))
      } else {
        ## This branch is used to keep the precomputed GUS operators, such as
        ## those set by MarkSample.
        waypoint$gus
      }

    input <-
      if (op == "Join")
        list(Clean(waypoint$x), Clean(waypoint$y))
      else if (op %in% c("GLA", "GF", "Filter", "Generated"))
        list(Clean(waypoint$data))
      else
        list()
    list(op = op, branch = branch, alias = alias, gus = gus, input = input)
  }

  gus <- ComputeGUS(Clean(x), Clean(y))

  keys <- substitute(keys)
  value1 <- substitute(value1)
  value2 <- substitute(value2)
  l <- length(convert.exprs(keys, x))

  if (l != log(length(gus$b), 2))
    stop("incorrect number of keys given: ", l)

  ## Constructing the function calls manually. A dummy name is used for the
  ## output of the summation to avoid name clashing.
  mean1 <- call("Sum", value1, substitute(junk))
  mean2 <- call("Sum", value2, substitute(junk))
  seed <- as.integer(runif(1) * 2^31)
  sample1 <- call("AdjBernoulli", keys, value1, seed = seed, min = min)
  sample2 <- call("AdjBernoulli", keys, value2, seed = seed, min = min)

  x <- do.call("Multiplexer", list(quote(x), mean1, sample1))
  y <- do.call("Multiplexer", list(quote(y), mean2, sample2))

  outputs <- convert.atts(substitute(outputs))

  GIST <- GIST(sampling::Covariance, a = gus$a, b = gus$b, foreign.key)
  Transition(GIST, outputs, list(x, y))
}

## Determine shared.
Shared <- function(data, shared, child = FALSE) {
  if (!is(data, "data"))
    return(shared)
  if (data$alias %in% shared)
    child = TRUE
  else if (child)
    shared <- c(shared, data$alias)
  unique(c(shared, unlist(lapply(data, Shared, shared, child), use.names = FALSE)))
}

## Given a waypoint, this returns a character vector containing the aliases of
## every waypoint upstream from it.
Parents <- function(data) {
  if (is(data, "data"))
    c(data$alias, unlist(lapply(data, Parents), use.names = FALSE))
}

ComputeGUS <- function(x, y) {
  x <- Compress(x)
  y <- Compress(y)

  ## This is used as a stack for BFS to get a list of nodes present.
  waypoints <- list()
  children <- list()
  parents <- list()
  stack <- list(x, y)

  while (length(stack) > 0) {
    ## Pop the stack
    x <- stack[[1]]
    stack <- stack[-1]

    ## Don't repeat already seen nodes.
    if (x$alias %in% names(parents))
      next
    else
      parents[[x$alias]] <- character()

    ## Add children to stack
    if (length(x$input) > 0) {
      for (input in x$input) {
        if (!input$alias %in% names(children))
          children[[input$alias]] <- character()
        parents[[x$alias]] <- c(parents[[x$alias]], input$alias)
        children[[input$alias]] <- c(children[[input$alias]], x$alias)
        stack <- c(stack, list(input))
      }
    }

    x$input <- NULL
    waypoints[[x$alias]] <- x
  }

  ## Top level nodes have not been given children information yet.
  children[setdiff(names(waypoints), names(children))] <- list(character())

  gus <- Simplify(waypoints, children, parents, list(Rule2, Rule3, Rule4, Rule5))
}

## Compresses GUSs that are in the same branch
Compress <- function(x) {
  if (x$op %in% c("Load", "GI")) {
    if (is.null(x$gus))
        x$gus <- list(a = 1, b = c(1, 1))
    c(x, finished = TRUE)
  } else {
    ## Everything other than a Load or a GI has input.
    x$input <- lapply(x$input, Compress)
    ## Compression was stopped before here. Just return the node.
    if (!all(sapply(x$input, `[[`, "finished")))
      return(c(x, finished = FALSE))
    ## Compression is different for a join because it takes 2 inputs.
    if (x$op != "Join") {
      ## Both the child and the parent must be in the same branch
      if (x$input[[1]]$branch != x$branch)
        return(c(x, finished = FALSE))
      if (is.null(x$gus)) ## Pass through case for Filter
        x$gus <- x$input[[1]]$gus
      else ## Combine GUS at waypoint for GLA, GF.
        x$gus <- mapply(`*`, x$gus, x$input[[1]]$gus)
      ## Throw away the inputs because they have been accounted for.
      x$input <- NULL
      ## The compressed result is returned as a leaf.
      c(x, finished = TRUE)
    } else {
      ## A join cannot be compressed if the inputs aren't in the same branch.
      if (length(unique(sapply(x$input, `[[`, "branch"))) != 1)
        return(c(x, finished = FALSE))
      x$gus <- list(a = x$input[[1]]$gus$a * x$input[[2]]$gus$a,
                    b = Combine(x$input[[2]]$gus$b, x$input[[1]]$gus$b))
      x$input <- NULL
      c(x, finished = TRUE)
    }
  }
}

## Used for combining two b vectors.
Combine <- function(x, y)
  as.numeric(matrix(x) %*% t(matrix(y)))

## Either of these functions are used to traverse the graph representing the
## query plan. Such a graph is described using three objects - a named list
## containing each node, a character mapping waypoint names to the names of
## their parents, and a character reverse mapping waypoint names to the names of
## their children.

## Given a waypoint name and the name of a parent to it, this returns the name
## of the other child to that parent.
Other <- function(parent, name, parents)
  setdiff(parents[[parent]], name)

## Each rule should take in a waypoint leaf, which will always be given as a
## shared waypoint (branch = 0), and the three objects used to describe the
## graph. It should return false if the leaf cannot be simplified; otherwise it
## should return a list containing a GUS element and a character vector of the
## names of the parents it will replace.

## TODO: The rules are labelled accordining to the notes Supriya sent me. They
## should be relabelled based on the published paper if it contains individual
## rules of simplification.

## TODO: These don't check information about branches. There should be checks to
## ensure that a non-tree query plans appears in a single branch. Currently, it
## would just get simplified and an error thrown later once repeated nodes are
## not properly replaced.
Rule2 <- function(leaf, waypoints, children, parents) {
  name <- leaf$alias
  parent.names <- children[[name]]
  if (length(parent.names) == 0)
    return(FALSE)
  parent.wps <- waypoints[parent.names]
  ops <- sapply(parent.wps, `[[`, "op")
  if (!all(ops == "Join"))
    return(FALSE)
  child.names <- unique(sapply(parent.names, Other, name, parents))
  if (length(child.names) == 1)
    return(FALSE)
  child.wps <- waypoints[child.names]
  if (!all(sapply(child.wps, `[[`, "finished")))
    return(FALSE)
  gus <- lapply(child.wps, `[[`, "gus")
  a.prod <- prod(sapply(gus, `[[`, "a"))
  gus <- leaf$gus
  print("applying rule 2")
  list(gus = list(a = a.prod * gus$b[[1]], b = a.prod * gus$b),
       parents = parent.names)
}

Rule3 <- function(leaf, waypoints, children, parents) {
  name <- leaf$alias
  parent.names <- children[[name]]
  if (length(parent.names) != 2)
    return(FALSE)
  parent.wps <- waypoints[parent.names]
  ops <- sapply(parent.wps, `[[`, "op")
  if (!all(ops == "Join"))
    return(FALSE)
  child.names <- unique(sapply(parent.names, Other, name, parents))
  if (length(child.names) > 1)
    return(FALSE)
  child <- waypoints[[child.names]]
  if (!child$finished)
    return(FALSE)
  print("applying rule 3")
  list(gus = list(a = leaf$gus$a * child$gus$a, b = Combine(leaf$gus$b, child$gus$b)),
       parents = parent.names)
}

## In the case that only one parent is a filter / GUS, we simulate a trivial GUS
## between the leaf and the other parent.
## This takes into account the case that the leaf is an input to the GIST.
Rule4 <- function(leaf, waypoints, children, parents) {
  name <- leaf$alias
  parent.names <- children[[name]]
  if (length(parent.names) == 0)
    return(FALSE)
  parent.wps <- waypoints[parent.names]
  ops <- sapply(parent.wps, `[[`, "op")
  ## At least one operator should be a GUS or Filter. We can fill in operators
  ## but it's pointless to fill in both of them.
  allowed.ops <- ops %in% c("GLA", "GF", "Filter")
  if (!any(allowed.ops))
    return(FALSE)
  ## These are the A coefficients: parameter 'a' for a GUS, 1 for a selection.
  A.coef <- function(wp, used)
    if (!used || is.null(wp$gus)) 1 else wp$gus$a
  A.prod <- prod(mapply(A.coef, parent.wps, allowed.ops))
  print("applying rule 4")
  list(gus = sapply(leaf$gus, `*`, A.prod),
       parents = parent.names[allowed.ops])
}

Rule5 <- function(leaf, waypoints, children, parents) {
  name <- leaf$alias
  parent.names <- children[[name]]
  if (length(parent.names) == 0)
    return(FALSE)
  parent.wps <- waypoints[parent.names]
  ops <- sapply(parent.wps, `[[`, "op")
  if (sum(ops == "Join") != 1)
    return(FALSE)
  print("applying rule 5")
  ## There are three cases, the first of which that the join is the only parent.
  ## This means that the leaf is used as one of the two input states. A trivial
  ## GUS is simulated.
  if (length(parent.names) == 1) {
    A.coef <- 1
    replace <- parent.names[ops == "Join"]
  } else {
    ## The parent that is not a join.
    other <- parent.wps[ops != "Join"][[1]]
    if (other$op %in% c("GLA", "GF", "Filter")) {
      ## The second case is that the other parent is valid.
      A.coef <- if (is.null(op$gus)) 1 else wp$gus$a
      replace <- parent.names
    } else{
      ## The third case is to simulate a trivial GUS between the leaf and the
      ## other parent because it isn't a filter or GUS.
      A.coef <- 1
      replace <- parent.names[ops == "Join"]
    }
  }
  ## The a coefficient for the join.
  a.coef <- waypoints[[Other(parent.names[ops == "Join"], name, parents)]]$gus$a
  list(gus = sapply(leaf$gus, `*`, A.coef * a.coef), parents = replace)
}

## This function takes in a list of waypoints and two characters mappings
## describing parent and children connectivity, which together describe a query
## plan, and a list of simplication rules.
Simplify <- function(waypoints, children, parents, rules) {
  repeat {
    ## This is part of breaking out of an outer loop.
    break.loop <- FALSE
    ## First we grab the shared waypoints that are also a leaf.
    leaves <- waypoints[names(parents)[sapply(parents, length) == 0]]
    leaves <- leaves[sapply(leaves, `[[`, "branch") == 0]
    for (rule in rules) {
      for (leaf in leaves) {
        result <- rule(leaf, waypoints, children, parents)
        if (!identical(FALSE, result)) {
          ## The rule was applied. Replace parents with new node.
          name <- leaf$alias
          wp <- list(op = "Simplified", branch = 0, alias = name,
                     gus = result$gus, finished = TRUE)
          parent.names <- result$parents
          ## The children of the waypoints being replaced are removed. The
          ## parent information of their parents is removed below when said
          ## parents are replaced.
          ## unlist is used because sapply will not combine the length-zero
          ## vectors that result when the parent's only child is the leaf.
          child.names <- unlist(sapply(parent.names, Other, name, parents, USE.NAMES = FALSE))
          waypoints[child.names] <- NULL
          parents[child.names] <- NULL
          children[child.names] <- NULL
          ## The waypoint is replaced by the simplified GUS.
          waypoints[parent.names] <- NULL
          waypoints[[name]] <- wp
          ## The new simplified GUS replaces the parent(s) of the old GUS.
          ## The parent map is modified by removing elements. There is no need
          ## to add an element for the new node because it is still a leaf and
          ## the name is kept.
          parents[parent.names] <- NULL
          ## Replace child mapping. The new node now has the combined parents of
          ## its parents. This shouldn't ever result in a node having more than
          ## 2 parents, hence the check.
          grandparents <- unlist(children[parent.names], use.names = FALSE)
          children[[name]] <- c(grandparents, setdiff(children[[name]], parent.names))
          if (length(children[[name]]) > 2)
            stop(name, " now has parents: ", paste(children[[name]], collapse = ""))
          ## The grandparents of the node must have their information changed so
          ## that they are parents to the new node. Altering the current element
          ## in a foreach loop does not actually change the container, so a
          ## counter must be maintained.
          counter <- 1
          for (parent in parents[grandparents]) {
            parent[parent %in% parent.names] <- name
            parents[grandparents][[counter]] <- parent
            counter <- counter + 1
          }
          children[parent.names] <- NULL
          break.loop <- TRUE
          break
        }
      }
      if (break.loop)
        break
    }
    if (break.loop)
      next
    else
      break
  }
  if (length(waypoints) != 1 || is.null(waypoints[[1]]$gus))
    stop ("Simplification failed.")
  waypoints[[1]]$gus
}
