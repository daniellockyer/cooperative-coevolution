
analyse <- function(filename) {
  data <- read.csv(file=filename, sep=",")
  mat <- as.matrix(data)
  avg <- apply(mat, 2, mean)
  plot(avg, type="l")
}

analyse("ga.csv")
analyse("ccga1.csv")
analyse("ccga1_clever.csv")
