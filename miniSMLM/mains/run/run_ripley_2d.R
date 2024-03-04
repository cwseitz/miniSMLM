library(spatstat)
library(ggplot2)
library(gridExtra)
library(dplyr)

prepareData <- function(file,nsample=3000,pixel_size=108.3,hw=5000){
  data <- read.csv(file)
  data = data[!duplicated(data$x_mle),]
  filtered_data <- data
  filtered_data <- filtered_data[sample(nrow(filtered_data), nsample), ]
  filtered_data$xclust <- filtered_data$y_mle * pixel_size #swap x and y
  filtered_data$yclust <- filtered_data$x_mle * pixel_size
  xcenter = mean(filtered_data$xclust)
  ycenter = mean(filtered_data$yclust)
  points <- ppp(filtered_data$xclust, filtered_data$yclust, owin(c(xcenter-hw,xcenter+hw), c(ycenter-hw,ycenter+hw)))
  points <- unique(points,rule="deldir")
  return(points)
}

computePCF <- function(points,rmax=700,numr=100){
  r = seq(0,rmax,length.out=numr)
  corr <- pcf(points,r=r)
  return(corr)
}

computeLFunction <- function(points,hw=5000,rmax=700) {
  L <- Lest(points, r_max=rmax, correction="best")
  L$iso <- L$iso - L$r
  return(L)
}

savePlot <- function(file,points){
  num_locations <- npoints(points)
  new_file <- gsub("\\.csv$", ".png", file)
  png(new_file)
  plot(points, main = paste(num_locations, "localizations"))
  dev.off()
  cat(paste("Plot saved as", new_file,"\n"))
}

dir <- '/home/cwseitz/Desktop/BRD4/STORM/240202/BD'
file_list <- list.files(path = dir, pattern = "\\.csv$", full.names = TRUE)
L_functions <- list()
combined <- data.frame(matrix(, nrow=513, ncol=0))

for (file in file_list) {
  points <- prepareData(file,nsample=3000)
  savePlot(file,points)
  L <- computeLFunction(points)
  corr <- computePCF(points)
  iso <- L$iso
  r <- L$r
  col_name <- tools::file_path_sans_ext(basename(file))
  combined[[col_name]] <- iso
  combined$r <- r
}

write.csv(combined, file = paste0(dir,"/combined_data.csv"), row.names = FALSE)



