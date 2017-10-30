data <- read.csv("data.csv")

plot(data$iteration, data$value, type="l", col="green")
par(new=T)
plot(data$iteration, data$fitness, type="l", col="red", axes=F, xlab=NA, ylab=NA)
axis(side = 4)