library(data.table)

#=== Load the data ============================================================================================

sales <- fread("https://raw.githubusercontent.com/ben519/sizecurve/master/sales.csv")

#=== Solver ============================================================================================

get_size_curve <- function(pairs, alpha = 0.0001, iters = 100){
  # Given a data.table of unique variant pairs (v1, v2) and associated
  # pairwise sales (n1, n2), return the most likely size curve
  
  # Copy input data.table
  pairs2 <- copy(pairs)
  
  # Insert total sales
  pairs2[, n := n1 + n2]
  
  # initialize probabilities for every variant
  variants <- data.table(variant = sort(unique(c(pairs2$v1, pairs2$v2))))
  variants[, p := 1/.N]
  
  # Gradient ascent
  for(i in seq_len(iters)){
    
    # Insert probabilities into pairs2
    pairs2[variants, p1 := i.p, on = c("v1"="variant")]
    pairs2[variants, p2 := i.p, on = c("v2"="variant")]
    
    # Calculate log likelihood
    pairs2[, p12 := p1/(p1 + p2)]
    pairs2[, logl := n1*log(p12) + (n - n1)*log(1 - p12)]
    logl <- sum(pairs2$logl)
    
    # Print
    print(paste0("iter: ", i, ", log likelihood: ", logl))
    
    # Calculate the gradient (partial log likelihood / partial p_i for every class)
    g1 <- pairs2[, list(grad = sum((n1/p12 - (n - n1)/(1 - p12)) * (p2)/(p1 + p2)^2)), keyby = list(variant = v1)]
    g2 <- pairs2[, list(grad = sum((n1/p12 - (n - n1)/(1 - p12)) * (-p1)/(p1 + p2)^2)), keyby = list(variant = v2)]
    grads <- rbind(g1, g2)[, list(grad = sum(grad)), keyby = variant]
    
    # Update probabilities (gradient step)
    variants[grads, grad := i.grad, on = "variant"]
    variants[, p := p + alpha * grad] 
    variants[, p := p/sum(p)]  # normalize
  }
  
  # Return
  return(variants[, list(variant, p)])
}

#=== Prep the data ============================================================================================

# Generate pairwise data
variants <- sort(unique(sales$variant))
pairs <- CJ(v1 = variants, v2 = variants)[v1 < v2]

# Count the sales per variant, for each pair, on days without depletion
pairs[, joinvar := v1]
t1 <- sales[pairs, on = c("variant"="joinvar")]
pairs[, joinvar := v2]
t2 <- sales[pairs, on = c("variant"="joinvar")]
combo <- rbind(t1, t2, use.names=TRUE)
combo <- combo[depleted == FALSE]
combo[, N := .N, by = list(v1, v2, date)]
combo <- combo[N == 2]
pairs <- combo[, list(
  n1 = sum(sales[v1 == variant]),
  n2 = sum(sales[v2 == variant])
), keyby = list(v1, v2)]

#=== Run ============================================================================================

sc <- get_size_curve(pairs, alpha = 0.0001, iters = 100)
print(sc)
