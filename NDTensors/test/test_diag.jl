@eval module $(gensym())
using Adapt: adapt
using GPUArraysCore: @allowscalar
using LinearAlgebra: diagm, dot, norm
using NDTensors:
    NDTensors,
    Dense,
    Diag,
    DiagTensor,
    Tensor,
    array,
    contract,
    data,
    dense,
    diaglength,
    diagindices,
    matrix,
    randomTensor,
    tensor
using Test: @testset, @test, @test_throws
include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: devices_list, is_supported_eltype

@testset "DiagTensor basic functionality" begin
    @testset "test device: $dev" for dev in devices_list(copy(ARGS)),
            elt in (Float32, ComplexF32, Float64, ComplexF64)

        if !is_supported_eltype(dev, elt)
            # Metal doesn't support double precision
            continue
        end
        t = dev(tensor(Diag(rand(elt, 100)), (100, 100)))
        @test conj(data(t)) == data(conj(t))
        @test typeof(conj(t)) <: DiagTensor

        d = rand(real(elt), 10)
        D = dev(Diag{elt}(d))
        @test eltype(D) == elt
        @test @allowscalar dev(Array(dense(D))) == convert.(elt, d)
        simD = similar(D)
        @test length(simD) == length(D)
        @test eltype(simD) == eltype(D)
        D = dev(Diag(one(elt)))
        @test eltype(D) == elt
        @test complex(D) == Diag(one(complex(elt)))
        @test similar(D) == Diag(0.0)

        D = Tensor(Diag(1), (2, 2))
        @test norm(D) == √2
        d = 3
        ## TODO this fails because uniform diag tensors are immutable
        #S = NDTensors.map_diag((i->i * 2), dev(D))
        # @allowscalar for i in 1:diaglength(S)
        #   @test  S[i,i] == 2.0 * D[i,i]
        # end

        vr = rand(elt, d)
        D = dev(tensor(Diag(vr), (d, d)))
        Da = array(D)
        Dm = matrix(D)
        Dp = permutedims(D, (2, 1))
        for x in (Da, Dm, Dp)
            @test x == dev(diagm(0 => vr))
            @test x == dev(diagm(0 => vr))
            @test x == D
        end
        @test sqrt(contract(D, (-1, -2), conj(D), (-1, -2))[]) ≈ norm(D)
        # This if statement corresponds to the reported bug:
        # https://github.com/JuliaGPU/Metal.jl/issues/364
        if !(dev == NDTensors.mtl && elt === ComplexF32)
            S = permutedims(dev(D), (1, 2), sqrt)
            @allowscalar begin
                for i in 1:diaglength(S)
                    @test S[i, i] ≈ sqrt(D[i, i])
                end
            end
        end
        S = NDTensors.map_diag(i -> 2 * i, dev(D))
        @allowscalar for i in 1:diaglength(S)
            @test S[i, i] == 2 * D[i, i]
        end

        a = dev(tensor(Dense(randn(elt, 3)), (3,)))
        @test diagindices(a) == 1:1:3
        a = dev(tensor(Dense(randn(elt, 9)), (3, 3)))
        @test diagindices(a) == 1:4:9
        a = dev(tensor(Dense(randn(elt, 36)), (3, 4, 3)))
        @test diagindices(a) == 1:16:33
        a = dev(tensor(Dense(randn(elt, 0)), (3, 0)))
        @test diagindices(a) == 1:1:0

        # Regression test for https://github.com/ITensor/ITensors.jl/issues/1199
        S = dev(tensor(Diag(randn(elt, 2)), (2, 2)))
        ## This was creating a `Dense{ReshapedArray{Adjoint{Matrix}}}` which, in mul!, was
        ## becoming a Transpose{ReshapedArray{Adjoint{Matrix}}} which was causing issues on
        ## dispatching GPU mul!
        V = dev(tensor(Dense(randn(elt, 12, 2)'), (3, 4, 2)))
        S1 = contract(S, (2, -1), V, (3, 4, -1))
        S2 = contract(dense(S), (2, -1), copy(V), (3, 4, -1))
        @test @allowscalar S1 ≈ S2
    end
end
@testset "DiagTensor contractions" for dev in devices_list(copy(ARGS))
    ## TODO add more GPU tests
    elt = (dev == NDTensors.mtl ? Float32 : Float64)
    t = dev(tensor(Diag(elt[1.0, 1.0, 1.0]), (3, 3)))
    A = dev(randomTensor(Dense{elt}, (3, 3)))

    @test sum(t) ≈ sum(array(t))
    @test sum(A) ≈ sum(array(A))
    @test prod(t) ≈ prod(array(t))
    @test prod(A) ≈ prod(array(A))

    @test contract(t, (1, -2), t, (-2, 3)) == t
    @test contract(A, (1, -2), t, (-2, 3)) == A
    @test contract(A, (-2, 1), t, (-2, 3)) == transpose(A)

    ## Testing sparse contractions on GPU
    t = dev(tensor(Diag(one(elt)), (3, 3)))
    @test contract(t, (-1, -2), A, (-1, -2))[] ≈ dot(dev(array(t)), array(A)) rtol = sqrt(
        eps(elt)
    )

    ## Test dot on GPU
    @test dot(t, A) ≈ dot(dev(array(t)), array(A)) rtol = sqrt(eps(elt))
end
nothing
end
