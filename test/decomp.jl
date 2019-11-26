using ITensors, Test

a = [-0.1, -0.12]
@test ITensors.truncate!(a) == (0., 0.)
@test length(a) == 1
a = [0.1, 0.01, 1e-13]
@test ITensors.truncate!(a,absoluteCutoff=true,cutoff=1e-5) == (1e-13, (0.01 + 1e-13)/2)
@test length(a) == 2

i = Index(2,"i")
j = Index(2,"j")
A = randomITensor(i,j)
@test_throws ArgumentError factorize(A, i, dir="fakedir")
