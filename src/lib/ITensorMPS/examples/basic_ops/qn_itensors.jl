using ITensors

d = 1
i = Index([QN(0, 2) => d, QN(1, 2) => d], "i")

# Parity conserving ITensors
# By default they have flux 0
@show A = randomITensor(i', dag(i))
println()
@show B = randomITensor(i', dag(i))
println()
@show C = randomITensor(QN(1, 2), i', dag(i))
println()

# Add them
@show A + B
println()

# Can't add QN ITensors with different flux
#A + C

# Contract them
@show A' * B
println()
@show A' * C
println()

# Combine the indices to turn into a vector
comb = combiner(i', dag(i))
@show A * comb
println()
