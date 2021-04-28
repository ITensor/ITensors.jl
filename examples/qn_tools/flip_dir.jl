using ITensors

# This is a short code showing how a combiner
# can be used to "flip" the direction of an Index
i = Index([QN(0) => 2, QN(1) => 3], "i")
j = Index([QN(0) => 2, QN(1) => 3], "j")
A = randomITensor(i, dag(j))
C = combiner(dag(j); tags="jflip", dir=-dir(dag(j)))
@show inds(A)
@show inds(A * C)
