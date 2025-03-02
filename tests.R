# 1. Install required R packages by running:
#    install.packages("bnlearn")
#    install.packages("gRain")

# 2. Run the R script:
#    source("test.R")

#    This will check d-separation and perform exact and approximate inference using Gibbs sampling.

# Load required packages
if (!require("bnlearn")) install.packages("bnlearn", dependencies=TRUE)
if (!require("gRain")) install.packages("gRain", dependencies=TRUE)

library(bnlearn)
library(gRain)

# Define the structure of the network (Burglary, Earthquake, Alarm, JohnCalls, MaryCalls)
net <- model2network("[Burglary][Earthquake][Alarm|Burglary:Earthquake][JohnCalls|Alarm][MaryCalls|Alarm]")

# Define the Conditional Probability Tables (CPTs)
cpt_burglary <- matrix(c(0.999, 0.001), ncol = 2, dimnames = list(NULL, c("False", "True")))
cpt_earthquake <- matrix(c(0.998, 0.002), ncol = 2, dimnames = list(NULL, c("False", "True")))
cpt_alarm <- array(c(0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999), 
                   dim = c(2, 2, 2),
                   dimnames = list("Alarm" = c("False", "True"), 
                                   "Burglary" = c("False", "True"), 
                                   "Earthquake" = c("False", "True")))
cpt_johncalls <- matrix(c(0.95, 0.05, 0.9, 0.1), ncol = 2, 
                        dimnames = list("JohnCalls" = c("False", "True"), "Alarm" = c("False", "True")))
cpt_marycalls <- matrix(c(0.99, 0.01, 0.7, 0.3), ncol = 2, 
                        dimnames = list("MaryCalls" = c("False", "True"), "Alarm" = c("False", "True")))

# Combine the CPTs into a list
cpt_list <- list(
  Burglary = cpt_burglary,
  Earthquake = cpt_earthquake,
  Alarm = cpt_alarm,
  JohnCalls = cpt_johncalls,
  MaryCalls = cpt_marycalls
)

# Fit the Bayesian Network with the CPTs
cfit <- custom.fit(net, dist = cpt_list)

# Step 1: Perform D-separation Test for "Burglary ⊥ MaryCalls | Alarm"
dsep_test <- dsep(net, x = "Burglary", y = "MaryCalls", z = "Alarm")
print(paste("D-separation test for Burglary ⊥ MaryCalls | Alarm:", dsep_test))

dsep_test <- dsep(net, x = "Burglary", y = "JohnCalls", z = "Alarm")
print(paste("D-separation test for Burglary ⊥ JohnCalls | Alarm:", dsep_test))

dsep_test <- dsep(net, x = "Burglary", y = "Earthquake", z = "Alarm")
print(paste("D-separation test for Burglary ⊥ Earthquake | Alarm:", dsep_test))

# Step 2: Perform Exact Inference for P(Alarm | JohnCalls=1, MaryCalls=1)
evidence <- list(JohnCalls = "True", MaryCalls = "True")
exact_inference_result <- querygrain(as.grain(cfit), nodes = "Alarm", evidence = evidence)
print("Exact Inference (P(Alarm | JohnCalls=1, MaryCalls=1)):")
print(exact_inference_result)

# Step 3: Approximate Inference using Gibbs Sampling (using gRain)
# For approximate inference, we use a simpler form of Gibbs sampling in R
# You can use the bnlearn::cpquery function for approximate inference

# In R, approximate inference is typically done using the "cpquery" function in bnlearn
approx_inference_result <- cpquery(cfit, event = (Alarm == "True"), 
                                   evidence = (JohnCalls == "True" & MaryCalls == "True"), n = 10000)
print("Approximate Inference (Gibbs Sampling result):")
print(approx_inference_result)

# In R, approximate inference is typically done using the "cpquery" function in bnlearn
approx_inference_result <- cpquery(cfit, event = (Burglary == "True"), 
                                   evidence = (JohnCalls == "True" & MaryCalls == "True"), n = 10000)
print("Approximate Inference (Gibbs Sampling result):")
print(approx_inference_result)

# Step 4: Extract Adjacency Matrix (A matrix) from the network
adj_mat <- amat(net)
print("Adjacency Matrix:")
print(adj_mat)

# Step 5: Save the adjacency matrix and CPTs if needed
write.csv(adj_mat, "adjacency_matrix.csv")
saveRDS(cpt_list, "cpt_list.rds")
