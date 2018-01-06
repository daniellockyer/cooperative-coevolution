library(ggplot2)

analyse <- function(filename, title) {
  data <- read.csv(file="ga.csv", sep=",")
  mat <- as.matrix(data)
  
  m1 <- mat[-1]
  row.names(m1) <- mat[,1]
  
  avg <- apply(m1, 2, mean)
  avg = na.omit(avg)
  return(avg)
}

ga <- analyse("ga.csv")
ccga1 <- analyse("ccga1.csv")
#ccga3 <- analyse("ccga3.csv")

#ymin <- min(ga, ccga1, ccga3)
#ymax <- max(ga, ccga1, ccga3)

ymin <- min(ga, ccga1)
ymax <- max(ga, ccga1)

plot(ga, type="l", col="blue", ylim=c(0, ymax), xaxs="i", xlab="Function Evaluations", ylab="Best Individual")
par(new=T)
plot(ccga1, type="l", col="red", ylim=c(0, ymax), xaxs="i", xaxt='n', yaxt='n', xlab="", ylab="")
#par(new=T)
#plot(ccga3, type="l", col="green", ylim=c(0, ymax), xaxs="i", xaxt='n', yaxt='n', xlab="", ylab="")
legend("topright",
       lty=1, cex=0.8,
       c("GA", "CCGA1", "CCGA3"),
       col=c("blue", "red", "green"))


