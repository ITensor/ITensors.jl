@eval module $(gensym())
using ITensors: Index, itensor
using LinearAlgebra: qr, svd
using MappedArrays: mappedarray
using Test: @test, @testset
f(i::Int...) = float(sum(iⱼ -> iⱼ^2, i))
f(i::CartesianIndex) = f(Tuple(i)...)
@testset "NDTensorsMappedArraysExt" begin
    a = mappedarray(f, CartesianIndices((2, 2)))
    b = copy(a)
    i, j = Index.((2, 2))
    ta = itensor(a, i, j)
    tb = itensor(b, i, j)
    @test ta ≈ tb
    @test ta[i => 1, j => 2] ≈ tb[i => 1, j => 2]
    @test 2 * ta ≈ 2 * tb
    @test ta + ta ≈ tb + tb
    @test ta * ta ≈ tb * tb
    ua, sa, va = svd(ta, i)
    @test ua * sa * va ≈ ta
    qa, ra = qr(ta, i)
    @test qa * ra ≈ ta
end
end
