using ITensors

# Define indices
a = Index(2, "a")
b = Index(2, "b")
c = Index(2, "c")

# Note that Index tags don't have 
# to match variable names
i = Index(3, "Red,Link")

# Define 3 order 2 tensors (matrices)
Z = ITensor(a, b)
X = ITensor(b, c)
Y = ITensor(b, c)

# Set some elements
Z[a => 1, b => 1] = 1.0
Z[a => 2, b => 2] = -1.0

X[b => 1, c => 2] = 1.0
X[b => 2, c => 1] = 1.0

Y[b => 1, c => 1] = 1.0
Y[b => 2, c => 2] = 1.0

# Operations with tensors
R = Z * X

S = Y + X

T = Y - X

# Print results
println("Z =\n", Z, "\n")
println("X =\n", X, "\n")
println("Y =\n", Y, "\n")
println("R = Z * X =\n", R, "\n")
println("S = Y + X =\n", S, "\n")
println("S = Y - X =\n", T, "\n")

# Check that adding incompatible tensors cause an error
try
  U = Z + X
catch
  println("Cannot add Z and X")
end

# Compare calculations to Julia arrays
jZ = [
  1.0 0.0
  0.0 -1.0
]
jX = [
  0.0 1.0
  1.0 0.0
]
jY = [
  1.0 0.0
  0.0 1.0
]
@assert Array(R, a, c) == jZ * jX
@assert Array(S, b, c) == jY + jX
@assert Array(T, b, c) == jY - jX
