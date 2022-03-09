using ITensors

N = 10
s = siteinds("Qubit", N)

# Make all of the operators
X = ops(s, [("X", n) for n in 1:N])
H = ops(s, [("H", n) for n in 1:N])
CX = ops(s, [("CX", n, m) for n in 1:N, m in 1:N])

# Start with the state |0000...⟩
ψ0 = productMPS(s, "0")

# Change to the state |1010...⟩
gates = [X[n] for n in 1:2:N]
ψ = apply(gates, ψ0; cutoff=1e-15)
@assert inner(ψ, productMPS(s, n -> isodd(n) ? "1" : "0")) ≈ 1

# Change to the state |10111011...⟩
append!(gates, [CX[n, n + 3] for n in 1:4:(N - 3)])
ψ = apply(gates, ψ0; cutoff=1e-15)
@assert inner(ψ, productMPS(s, ["1", "0", "1", "1", "1", "0", "1", "1", "1", "0"])) ≈ 1

# Change the state |10111011...⟩ to the (|+⟩, |-⟩) basis
append!(gates, [H[n] for n in 1:N])
ψ = apply(gates, ψ0; cutoff=1e-15)
