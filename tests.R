# Load required packages
library(bnlearn)
library(gRain)

# Step 1: Define the Bayesian Network structure and model
net <- model2network("[Burglary][Earthquake][Alarm|Burglary:Earthquake][JohnCalls|Alarm][MaryCalls|Alarm]")
graphviz.plot(net)

# Step 2: Define the Conditional Probability Tables (CPTs)
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

# Step 3: Combine all CPTs into a list
cpt_list <- list(
  Burglary = cpt_burglary,
  Earthquake = cpt_earthquake,
  Alarm = cpt_alarm,
  JohnCalls = cpt_johncalls,
  MaryCalls = cpt_marycalls
)

# Step 4: Fit the Bayesian Network with the CPTs
cfit <- custom.fit(net, dist = cpt_list)

# Step 5: Convert the model to a grain object and compile it
bn.grain <- as.grain(cfit)
bn.grain <- compile(bn.grain)

# Step 6: Define evidence for exact inference
evidence <- list(JohnCalls = "True", MaryCalls = "True")

# Step 7: Perform Exact Inference using gRain
query_result <- querygrain(bn.grain, nodes = "Alarm", evidence = evidence)
print("Exact Inference (gRain):")
print(query_result)

# Step 8: Perform Approximate Inference using Gibbs Sampling
# Assuming you have a Python script that you can call via system or use R's RPython package

# Print or compare the results manually or programmatically
