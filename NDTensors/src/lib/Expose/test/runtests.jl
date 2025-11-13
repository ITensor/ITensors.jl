@eval module $(gensym())
using GPUArraysCore: @allowscalar
using LinearAlgebra:
    LinearAlgebra,
    Adjoint,
    Diagonal,
    Hermitian,
    Symmetric,
    Transpose,
    eigen,
    mul!,
    norm,
    qr,
    svd
using NDTensors: NDTensors, mul!!
using NDTensors.Expose: Expose, Exposed, expose
using NDTensors.GPUArraysCoreExtensions: cpu
using StableRNGs: StableRNG
using Test: @testset, @test, @test_broken
include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list

@testset "Testing Expose $dev, $elt" for dev in devices_list(ARGS),
        elt in (Float32, ComplexF32)

    rng = StableRNG(1234)
    v = dev(randn(rng, elt, 10))
    vt = transpose(v)
    va = v'

    E = expose(v)
    @test any(>(0) ∘ real, E)

    Et = expose(vt)
    Ea = expose(va)
    v_type = typeof(v)
    e_type = eltype(v)
    @test typeof(E) == Exposed{v_type, v_type}
    @test typeof(Et) == Exposed{v_type, Transpose{e_type, v_type}}
    @test typeof(Ea) == Exposed{v_type, Adjoint{e_type, v_type}}

    @test parent(E) == v
    @test parent(Et) == v
    @test parent(Ea) == v
    @test transpose(E) == vt
    @test cpu(E) == cpu(v)
    @test cpu(Et) == cpu(vt)

    m = reshape(v, (5, 2))
    mt = transpose(m)
    ma = m'
    E = expose(m)
    Et = expose(mt)
    Ea = expose(ma)

    m_type = typeof(m)
    @test typeof(E) == Exposed{m_type, m_type}
    @test typeof(Et) == Exposed{m_type, Transpose{e_type, m_type}}
    @test typeof(Ea) == Exposed{m_type, Adjoint{e_type, m_type}}

    o = dev(randn(elt, 1))
    expose(o)[] = 2
    @test expose(o)[] == 2

    fill!(m, zero(elt))
    @test any(!Base.isinf, expose(m))

    mp = copy(Ea)
    @test mp == ma
    fill!(ma, elt(2))
    copyto!(expose(mp), expose(ma))
    @test mp == ma

    q, r = qr(expose(mp))
    @test q * r ≈ mp

    q, r = Expose.qr_positive(expose(mp))
    @test q * r ≈ mp

    square = dev(rand(real(elt), (10, 10)))
    square = (square + transpose(square)) / 2
    ## CUDA only supports Hermitian or Symmetric eigen decompositions
    ## So I symmetrize square and call symetric here
    l, U = eigen(expose(Symmetric(square)))
    @test eltype(l) == real(elt)
    @test eltype(U) == real(elt)
    @test square * U ≈ U * Diagonal(l)

    square = dev(rand(elt, (10, 10)))
    # Can use `hermitianpart` in Julia 1.10
    square = (square + square') / 2
    ## CUDA only supports Hermitian or Symmetric eigen decompositions
    ## So I symmetrize square and call symetric here
    l, U = eigen(expose(Hermitian(square)))
    @test eltype(l) == real(elt)
    @test eltype(U) == elt
    @test square * U ≈ U * Diagonal(l)

    U, S, V, = svd(expose(mp))
    @test eltype(U) == elt
    @test eltype(S) == real(elt)
    @test eltype(V) == elt
    @test U * Diagonal(S) * V' ≈ mp

    cm = dev(randn(elt, 2, 2))
    mul!(expose(cm), expose(mp), expose(mp'), 1.0, 0.0)
    @test cm ≈ mp * mp'

    @test permutedims(expose(mp), (2, 1)) == transpose(mp)
    fill!(mt, 3)
    permutedims!(expose(m), expose(mt), (2, 1))
    @test norm(m) ≈ sqrt(3^2 * 10)
    @test size(m) == (5, 2)
    permutedims!(expose(m), expose(mt), (2, 1), +)
    @test size(m) == (5, 2)
    @test norm(m) ≈ sqrt(6^2 * 10)

    m = reshape(m, (5, 2, 1))
    mt = fill!(similar(m), elt(3))
    m = permutedims(expose(m), (2, 1, 3))
    @test size(m) == (2, 5, 1)
    permutedims!(expose(m), expose(mt), (2, 1, 3))
    @test norm(m) ≈ sqrt(3^2 * 10)
    permutedims!(expose(m), expose(mt), (2, 1, 3), -)
    @test norm(m) == 0

    x = dev(rand(elt, 4, 4))
    y = dev(rand(elt, 4, 4))
    copyto!(expose(y), expose(x))
    @test y == x

    y = dev(rand(elt, 4, 4))
    x = Base.ReshapedArray(dev(rand(elt, 16)), (4, 4), ())
    copyto!(expose(y), expose(x))
    @test cpu(y) == cpu(x)
    @test cpu(copy(expose(x))) == cpu(x)

    ## Tests for Metal because permutedims with ReshapedArray does not work properly
    ## transpose(ReshapedArray(MtlArray)) fails with scalar indexing so calling copy to
    ## evaluate tests in the following tests
    y = dev(rand(elt, 4, 4))
    @test permutedims(expose(y), (2, 1)) == transpose(y)
    y = Base.ReshapedArray(y, (2, 8), ())
    @test permutedims(expose(y), (2, 1)) == transpose(copy(expose(y)))
    yt = dev(rand(elt, (8, 2)))
    permutedims!(expose(y), expose(yt), (2, 1))
    @test copy(expose(y)) == transpose(yt)
    yt = dev(rand(elt, 8, 2))
    permutedims!(expose(yt), expose(y), (2, 1))
    @test copy(expose(y)) == transpose(yt)

    y = reshape(dev(randn(elt, 8))', 2, 4)
    x = Base.ReshapedArray(dev(randn(elt, 8, 8)'[1:8]), (2, 4), ())
    z = dev(fill!(Matrix{elt}(undef, (2, 4)), 0.0))
    for i in 1:2
        for j in 1:4
            @allowscalar z[i, j] = y[i, j] * x[i, j]
        end
    end
    permutedims!(expose(y), expose(x), (1, 2), *)
    @allowscalar @test reshape(z, size(y)) ≈ y
    for i in 1:2
        for j in 1:4
            @allowscalar z[i, j] = x[i, j] * y[i, j]
        end
    end
    permutedims!(expose(x), expose(y), (1, 2), *)
    ## I copy x here because it is a ReshapedArray{SubArray} which causes `≈`
    ## to throw an error
    @test z ≈ copy(expose(x))

    y = dev(rand(elt, 4, 4))
    x = @view dev(rand(elt, 8, 8))[1:4, 1:4]
    copyto!(expose(y), expose(x))
    @test y == x
    @test copy(x) == x

    y = dev(randn(elt, 16))
    x = reshape(dev(randn(elt, 4, 4))', 16)
    copyto!(expose(y), expose(x))
    @allowscalar begin
        @test y == x
        @test copy(x) == x
    end

    y = dev(randn(elt, 8))
    x = @view reshape(dev(randn(elt, 8, 8))', 64)[1:8]
    copyto!(expose(y), expose(x))
    @allowscalar begin
        @test y == x
        ## temporarily use expose copy because this is broken in Metal 1.1
        @test copy(expose(x)) == x
    end

    y = Base.ReshapedArray(dev(randn(elt, 16)), (4, 4), ())
    x = dev(randn(elt, 4, 4))
    permutedims!(expose(y), expose(x), (2, 1))
    @test cpu(y) == transpose(cpu(x))

    ##########################################
    ### Testing an issue with CUDA&Metal transpose/adjoint mul
    A = dev(randn(elt, (3, 2)))
    B = dev(randn(elt, (3, 4)))
    C = dev(randn(elt, (4, 2)))
    Cp = copy(C)

    ## This fails with scalar indexing
    if dev != cpu
        @test_broken mul!(transpose(C), transpose(A), B, true, false)
    end
    mul!(C, transpose(B), A, true, false)
    mul!(expose(transpose(Cp)), expose(transpose(A)), expose(B), true, false)
    @test C ≈ Cp
    Cp = zero(C)
    ## Try calling mul!! with transposes to verify that code works
    Cpt = mul!!(transpose(Cp), transpose(A), B, true, false)
    @test transpose(Cpt) ≈ C

    Cp = zero(C)
    ## This fails with scalar indexing
    if dev != cpu
        @test_broken mul!(C', A', B, true, false)
    end
    mul!(C, B', A, true, false)
    mul!(expose(Cp'), expose(A'), expose(B), true, false)
    @test C ≈ Cp
    Cp = zero(C)
    Cpt = mul!!(Cp', A', B, true, false)
    @test Cpt' ≈ C

    ##################################
    ### Add test for transpose(reshape(adjoint )) failure in CUDA

    A = dev(transpose(reshape(randn(elt, 2, 12)', (12, 2))))
    B = dev(randn(elt, 2, 2))
    C = dev(zeros(elt, 2, 12))
    mul!(expose(C), expose(B), expose(A), true, false)
    Cp = cpu(similar(C))
    mul!(expose(Cp), expose(cpu(B)), expose(cpu(A)), true, false)
    @test cpu(C) ≈ Cp
    zero(C)
    mul!!(C, B, A, true, false)
    @test cpu(C) ≈ Cp

    ##################################
    ### Add test for append! to address scalar indexing in GPUs
    ## For now, Metal doesn't have a `resize!` function so all the tests are failing
    if (dev == NDTensors.mtl)
        continue
    end
    A = dev(randn(elt, 10))
    Ap = copy(A)
    B = randn(elt, 3)
    C = append!(expose(A), B)

    @test length(C) == 13
    @test sum(C) ≈ sum(Ap) + sum(B)

    A = Ap
    B = dev(randn(elt, 29))
    Bp = copy(B)
    C = append!(expose(B), A)
    @test length(C) == 39
    @test sum(C) ≈ sum(Bp) + sum(Ap)
    @allowscalar for i in 1:length(B)
        C[i] == B[i]
    end
end
end
