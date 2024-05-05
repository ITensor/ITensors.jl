using ITensors
using ITensors.ITensorNetworkMaps
using KrylovKit
using LinearAlgebra

χ = 3
d = 2
l = Index(χ, "l")
s = Index(d, "s")

l0 = addtags(l, "c=0")
l1 = addtags(l, "c=1")
A = randomITensor(l0, l1, s)
A′ = prime(dag(A); inds=(l0, l1))

T = ITensorNetworkMap([A, A′]; input_inds=(l1, l1'), output_inds=(l0, l0'))
v = randomITensor(input_inds(T))
@show T(v)
@show T * v
@show (I - T)(v)
@show (3I - 2T)(v)

D, V = eigsolve(T, v)
@show (T - D[1] * I)(V[1])
