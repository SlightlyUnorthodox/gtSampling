.onAttach <- function(libname, pkgname) {
  ## The sampling Grokit library is added.
  grokit$libraries <- c(grokit$libraries, "sampling")

  ## The plugins for sampling are registered.
  label <- create.name("sampling", "plugin")

  ## If the sample estimation GLA is used, the plan is marked as a sample. Only
  ## sampling GLAs are allowed afterwards and any sampling information is passed
  ## through.
  register.plugin(label, "GLA", TRUE, function(waypoint, info) {
    info <- info[[label]]
    ## The above waypoint needs a sample, i.e. it was sampling::Estimate.
    if (isTRUE(info$is.estimate))
      return(list(needs.sample = TRUE))
    switch(get.function.name(waypoint$gla),
           "statistics::Reservoir_Sampling" = {
             ## Sampling without replacement case.
             size = waypoint$gla$args$size
             card = waypoint$gla$args$cardinality
             assert(!is.null(card),
                    "Cardinality not given for reservoir sampling.")
             list(is.sample = TRUE, a = size / card,
                  b = c(size * (size - 1) / card / (card - 1), size / card))
           },
           ## The default case. Other GLAs can't be used during sampling.
           assert(!isTRUE(info$needs.sample),
                  "Non-sampling GLA used in a sampling query.")
           )
  })

  ## The only GF allowed between sampling and estimate is a Bernoulli filter.
  register.plugin(label, "GF", TRUE, function(waypoint, info) {
    if (get.function.name(waypoint$gf) == "base::BernoulliSample") {
      p = waypoint$gf$args$p
      list(is.sample = TRUE, a = p, b = c(p * p, p))
    } else {
      assert(!isTRUE(info[[label]][["needs.sample"]]), "Illegal waypoint used.")
    }
  })

  ## Generators, GIs, GISTs, and Loads are not allowed in sampling queries.
  for (class in c("Generated", "GI", "GIST", "Load"))
    register.plugin(label, class, TRUE, function(waypoint, info) {
      assert(!isTRUE(info[[label]][["needs.sample"]]), "Illegal waypoint used.")
      if (is(waypoint, "GIST") && get.function.name(waypoint$gist) == "sampling::Estimate")
        list(is.estimate = TRUE)
    })

  ## Joins and Filters both just pass through that they need a sample.
  for (class in c("Join", "Filter"))
    register.plugin(label, class, TRUE, function(waypoint, info) {
      info[[label]]
    })

  ## Filters just pass their information up.
  register.plugin(label, "Filter", FALSE, function(waypoint, info) {
    waypoint$extra[[label]] <- waypoint$data$extra[[label]]
    waypoint
  })

  ## The Multiplexer immediately before the Estimate GIST must pass information though.
  register.plugin(label, "GLA", FALSE, function(waypoint, info) {
    name <- get.function.name(waypoint$gla)
    if (name == "base::Multiplexer") {
      waypoint$extra[[label]] <- waypoint$data$extra[[label]]
      waypoint
    } else {
      waypoint
    }
  })

  ## The sampling information is computed for a join.
  register.plugin(label, "Join", FALSE, function(waypoint, info) {
    ## If the info from above is marked as needing a sample, then we can assume
    ## that the two sides of the join are both a GUS. If they were not, one of
    ## the previous plugins would have thrown an error.
    if (isTRUE(info[[label]][["needs.sample"]])) {
      x <- waypoint$x$extra[[label]]
      y <- waypoint$y$extra[[label]]
      waypoint$extra[[label]] <- list(a = x$a * y$a,
                                      b = as.numeric(matrix(y$b) %*% t(matrix(x$b))))
    }
    waypoint
  })

  ## The sampling information is passed through to the template arguments.
  register.plugin(label, "GIST", FALSE, function(waypoint, info) {
    if (get.function.name(waypoint$gist) == "sampling::Estimate") {
      for (element in c("a", "b"))
        waypoint$gist$args[[element]] <- waypoint$states[[1]]$extra[[label]][[element]]
      waypoint
    } else {
      waypoint
    }
  })
}

