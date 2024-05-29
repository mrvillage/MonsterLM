# nolint start

#' @docType package
#' @usage NULL
#' @useDynLib monsterlm, .registration = TRUE
NULL

#' Return string `"Hello world!"` to R.
#' @export
hello_world <- function() .Call(wrap__hello_world)

# nolint end
