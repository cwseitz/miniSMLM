library(spatstat)
library(ggplot2)
library(gridExtra)
library(dplyr)

# Function to compute L-function and return it
computeLFunction <- function(file) {
  ############################
  # Load the dataset
  ############################
  
  # Read the CSV file
  data <- read.csv(file)
  new_file <- gsub("\\.csv$", ".png", file)
  filtered_data <- data %>%
    filter(N0 >= 10, N0 <= 5000, x_mle > 0, y_mle > 0)
  filtered_data = filtered_data[!duplicated(filtered_data$x_mle),]
  
  filtered_data <- filtered_data[sample(nrow(filtered_data), 10000), ] #sample N localizations to reduce compute load
  filtered_data$xclust <- filtered_data$y_mle * 108.3 #swap x and y for consistency with python code
  filtered_data$yclust <- filtered_data$x_mle * 108.3
  
  ############################
  # Point pattern statistics
  ############################
  
  # Create a point pattern object
  hw = 5000
  xcenter = mean(filtered_data$xclust)
  ycenter = mean(filtered_data$yclust)
  points <- ppp(filtered_data$xclust, filtered_data$yclust, owin(c(xcenter-hw,xcenter+hw), c(ycenter-hw,ycenter+hw)))
  num_locations <- npoints(points)
  png(new_file)
  y_min <- min(filtered_data$yclust)
  y_max <- max(filtered_data$yclust)
  plot(points, ylim = c(y_max, y_min), main = paste(num_locations, "localizations"))
  dev.off()
  cat(paste("Plot saved as", new_file,"\n"))
  points <- unique(points,rule="deldir")
  
  # Set the maximum distance for estimation
  r_max <- 500  # Set your desired maximum distance
  
  # Compute the L-function
  L <- Lest(points, rmax = r_max, correction = "none")
  L$un <- L$un - L$r
  return(L)  # Return the computed L-function
}

##################################################

dir <- '/home/cwseitz/Desktop/230823/Ctrl'
file_list <- list.files(path = dir, pattern = "\\.csv$", full.names = TRUE)

# Initialize a list to store L-functions
L_functions <- list()
combined <- data.frame(matrix(, nrow=513, ncol=0))

# Iterate over the file list, compute L-functions, and store them
for (file in file_list) {
  L <- computeLFunction(file)
  un <- L$un
  r <- L$r
  
  # Extract the base file name without extension
  col_name <- tools::file_path_sans_ext(basename(file))
  
  # Assign 'un' values to a column with the file name as the column name
  combined[[col_name]] <- un
  combined$r <- r
}

write.csv(combined, file = paste0(dir,"/combined_data.csv"), row.names = FALSE)



