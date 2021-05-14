using ITensors
using KrylovKit
using Random: seed!

seed!(1234)

struct TransferMatrix
  A::ITensor
end

function (T::TransferMatrix)(v::ITensor)
  A = addtags(T.A, "ket"; tags="Link")
  Adag = addtags(dag(T.A), "bra"; tags="Link")
  return noprime(A * v * Adag)
end

χ = 10
d = 2

l = Index(χ, "Link")
s = Index(d, "Site")

A = randomITensor(dag(l)', s, l)

T = TransferMatrix(A)
b = randomITensor(addtags(dag(l), "bra"), addtags(l, "ket"))
b += swaptags(dag(b), "bra", "ket")

# Solve Tx = b
x, con = linsolve(T, b; krylovdim=80, tol=1e-4)

@show con

@show norm(T(x) - b)
