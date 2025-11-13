@eval module $(gensym())
using NDTensors
using NDTensors: denseblocks
using NDTensors.MetalExtensions: mtl
using Test: @testset, @test, @test_throws, @test_broken
using GPUArraysCore: @allowscalar
include("NDTensorsTestUtils/NDTensorsTestUtils.jl")
using .NDTensorsTestUtils: devices_list

struct MyInd
    dim::Int
end
NDTensors.dim(i::MyInd) = i.dim

@testset "Dense Tensors" begin
    @testset "test device: $dev" for dev in devices_list(copy(ARGS))
        elt = dev == mtl ? Float32 : Float64
        # Testing with GPU and CPU backends
        @testset "DenseTensor basic functionality" begin
            A = dev(Tensor(elt, (3, 4)))
            @allowscalar for I in eachindex(A)
                @test A[I] == 0
            end

            @test @allowscalar A[2, 1] isa elt
            @test dims(A[1:2, 1]) == (2,)
            @test dims(A[1:2, 2]) == (2,)
            @test dims(A[2:3, 2]) == (2,)
            @test dims(A[2, 2:4]) == (3,)
            @test dims(A[2:3, 2:4]) == (2, 3)
            @test dims(A[2:3, 2:end]) == (2, 3)
            @test dims(A[3, 2:end]) == (3,)

            @test dense(A) ≡ A
            @test denseblocks(A) ≡ A

            randn!(A)

            @test ndims(A) == 2
            @test dims(A) == (3, 4)
            @test inds(A) == (3, 4)

            Aview = A[2:3, 2:3]
            @test dims(Aview) == (2, 2)
            ## Added for issue 1431 create a tensor from
            ## a sliced view of another tensor
            Acopy = Tensor(NDTensors.storage(Aview), (1, 4))
            @test NDTensors.cpu(data(Acopy)) == NDTensors.cpu(data(Aview))
            @test dims(Acopy) == (1, 4)

            B = dev(Tensor(elt, undef, (3, 4)))
            randn!(B)
            C = copy(A)
            C = permutedims!!(C, B, (1, 2), +)
            Cp = NDTensors.map_diag(i -> 2 * i, C)
            @allowscalar for i in 1:diaglength(Cp)
                @test Cp[i, i] == 2 * C[i, i]
            end

            Ap = permutedims(A, (2, 1))
            @allowscalar begin
                for I in eachindex(A)
                    @test A[I] != 0
                end

                for I in eachindex(A)
                    @test A[I] != 0
                end

                ## TODO Currently this fails with scalar indexing on CUDA
                ## Because A + B calls
                ## +(A::DenseTensor{Float64, 2, Tuple{Int64, Int64}, Dense{Float64, CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}}, B::DenseTensor{Float64, 2, Tuple{Int64, Int64}, Dense{Float64, CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}})
                ## @ Base ./arraymath.jl:8
                #C = A + B

                for I in eachindex(C)
                    @test C[I] == A[I] + B[I]
                end

                for I in eachindex(A)
                    @test A[I] == Ap[NDTensors.permute(I, (2, 1))]
                end

                A[1, 1] = 11
                @test A[1, 1] == 11

                @test A[2, 2] == Aview[1, 1]
            end

            ## Testing A .= α .* B .+ β .* A
            C = copy(A)
            @allowscalar fill!(B, zero(elt))
            β = elt(2)
            α = elt(1)
            permutedims!!(A, B, (1, 2), (a, b) -> +(*(β, a), *(α, b)))
            @allowscalar 2 .* C == A
            randn!(B)
            C = copy(A)
            A = permutedims!!(A, B, (1, 2), (a, b) -> +(*(β, a), *(α, b)))
            @allowscalar for i in 1:3, j in 1:4
                @test A[i, j] == α * B[i, j] + β * C[i, j]
            end

            ## add elt around 2.0 to preserve the eltype of A.
            @test data(A * elt(2.0)) == data(elt(2.0) * A)

            Asim = similar(data(A), 10)
            @test eltype(Asim) == elt
            @test length(Asim) == 10

            t = dev(Tensor(complex(elt), (100, 100)))
            randn!(t)
            @test conj(data(store(t))) == data(store(conj(t)))
            @test typeof(conj(t)) <: DenseTensor

            @test Dense(complex(elt)) == Dense{complex(elt)}()
            @test Dense(complex(elt)) == complex(Dense(elt))

            D = dev(Tensor(complex(elt), (100, 100)))
            @test eltype(D) == complex(elt)
            @test ndims(D) == 2
            @test dim(D) == 100^2

            E = dev(Tensor(complex(elt), undef, (100, 100)))
            @test eltype(E) == complex(elt)
            @test ndims(E) == 2
            @test dim(E) == 100^2

            F = dev(Tensor(elt, (100, 100)))
            @test eltype(F) == elt
            @test ndims(F) == 2
            @test dim(F) == 100^2

            G = dev(Tensor(elt, undef, (100, 100)))
            @test eltype(G) == elt
            @test ndims(G) == 2
            @test dim(G) == 100^2

            H = dev(Tensor(complex(elt), undef, (100, 100)))
            @test eltype(H) == complex(elt)
            @test ndims(H) == 2
            @test dim(H) == 100^2

            I_arr = dev(rand(elt, 10, 10, 10))
            I = dev(Tensor(I_arr, (10, 10, 10)))
            @test eltype(I) == elt
            @test dim(I) == 1000
            @test Array(I) == I_arr

            J = dev(Tensor(elt, (2, 2)))
            K = dev(Tensor(elt, (2, 2)))
            @test Array(J * K) ≈ Array(J) * Array(K)
        end

        @testset "Random constructor" begin
            T = dev(randomTensor(elt, (2, 2)))
            @test dims(T) == (2, 2)
            @test eltype(T) == elt
            @test @allowscalar T[1, 1] ≉ 0
            @test norm(T) ≉ 0

            Tc = dev(randomTensor(complex(elt), (2, 2)))
            @test dims(Tc) == (2, 2)
            @test eltype(Tc) == complex(elt)
            @test @allowscalar Tc[1, 1] ≉ 0
            @test norm(Tc) ≉ 0
        end

        @testset "Complex Valued Tensors" begin
            d1, d2, d3 = 2, 3, 4
            T = dev(randomTensor(complex(elt), (d1, d2, d3)))

            rT = real(T)
            iT = imag(T)
            cT = conj(T)

            @allowscalar for n1 in 1:d1, n2 in 1:d2, n3 in 1:d3
                @test rT[n1, n2, n3] ≈ real(T[n1, n2, n3])
                @test iT[n1, n2, n3] ≈ imag(T[n1, n2, n3])
                @test cT[n1, n2, n3] ≈ conj(T[n1, n2, n3])
            end
        end

        @testset "Custom inds types" begin
            T = dev(Tensor(elt, (MyInd(2), MyInd(3), MyInd(4))))
            @test store(T) isa Dense
            @test eltype(T) == elt
            @test norm(T) == 0
            @test dims(T) == (2, 3, 4)
            @test ndims(T) == 3
            @test inds(T) == (MyInd(2), MyInd(3), MyInd(4))
            @allowscalar begin
                T[2, 1, 2] = 1.21
                @test T[2, 1, 2] == elt(1.21)
            end
            @test norm(T) == elt(1.21)

            T = dev(randomTensor(complex(elt), (MyInd(4), MyInd(3))))
            @test store(T) isa Dense
            @test eltype(T) == complex(elt)
            @test norm(T) > 0
            @test dims(T) == (4, 3)
            @test ndims(T) == 2
            @test inds(T) == (MyInd(4), MyInd(3))

            T2 = 2 * T
            @test eltype(T2) == complex(elt)
            @test store(T2) isa Dense
            @test norm(T2) > 0
            @test norm(T2) / norm(T) ≈ 2
            @test dims(T2) == (4, 3)
            @test ndims(T2) == 2
            @test inds(T2) == (MyInd(4), MyInd(3))
        end

        @testset "generic contraction" begin
            # correctness of _gemm!
            for alpha in [0.0, 1.0, 2.0]
                for beta in [0.0, 1.0, 2.0]
                    for tA in ['N', 'T']
                        for tB in ['N', 'T']
                            A = randn(4, 4)
                            B = randn(4, 4)
                            C = randn(4, 4)
                            A = BigFloat.(A)
                            B = BigFloat.(B)
                            C2 = BigFloat.(C)
                            NDTensors._gemm!(tA, tB, alpha, A, B, beta, C)
                            NDTensors._gemm!(tA, tB, alpha, A, B, beta, C2)
                            @test C ≈ C2
                        end
                    end
                end
            end
        end

        @testset "Contraction with size 1 block and NaN" begin
            @testset "No permutation" begin
                R = dev(Tensor(complex(elt), (2, 2, 1)))
                fill!(R, elt(NaN))
                @test @allowscalar any(isnan, R)
                T1 = dev(randomTensor(elt, (2, 2, 1)))
                T2 = dev(randomTensor(complex(elt), (1, 1)))
                NDTensors.contract!(R, (1, 2, 3), T1, (1, 2, -1), T2, (-1, 1))
                @test @allowscalar !any(isnan, R)
                @test convert(Array, R) ≈ convert(Array, T1) * T2[]
            end

            @testset "Permutation" begin
                R = dev(Tensor(complex(elt), (2, 2, 1)))
                fill!(R, elt(NaN))
                @test @allowscalar any(isnan, R)
                T1 = dev(randomTensor(elt, (2, 2, 1)))
                T2 = dev(randomTensor(complex(elt), (1, 1)))
                NDTensors.contract!(R, (2, 1, 3), T1, (1, 2, -1), T2, (-1, 1))
                @test @allowscalar !any(isnan, R)
                @test convert(Array, R) ≈ permutedims(convert(Array, T1), (2, 1, 3)) * T2[]
            end
        end
    end

    # Only CPU backend testing
    @testset "Contract with exotic types" begin
        # BigFloat is not supported on GPU
        ## randn(BigFloat, ...) is not defined in Julia 1.6
        a = BigFloat.(randn(Float64, 2, 3))
        t = Tensor(a, (1, 2, 3))
        m = Tensor(a, (2, 3))
        v = Tensor([one(BigFloat)], (1,))

        @test m ≈ contract(t, (-1, 2, 3), v, (-1,))
        tp = similar(t)
        NDTensors.contract!(tp, (1, 2, 3), t, (1, 2, 3), v, (1,), false, false)
        @test iszero(tp)

        fill!(tp, one(BigFloat))
        NDTensors.contract!(tp, (1, 2, 3), t, (1, 2, 3), v, (1,), false, true)
        for i in tp
            @test i == one(BigFloat)
        end

        rand_factor = BigFloat(randn(Float64))
        NDTensors.contract!(tp, (1, 2, 3), t, (1, 2, 3), v, (1,), false, rand_factor)
        for i in tp
            @test i == rand_factor
        end
    end

    @testset "change backends" begin
        a, b, c = [randn(5, 5) for i in 1:3]
        backend_auto()
        @test NDTensors.gemm_backend[] == :Auto
        @test NDTensors.auto_select_backend(typeof.((a, b, c))...) ==
            NDTensors.GemmBackend(:BLAS)
        res1 = NDTensors._gemm!('N', 'N', 2.0, a, b, 0.2, copy(c))
        backend_blas()
        @test NDTensors.gemm_backend[] == :BLAS
        res2 = NDTensors._gemm!('N', 'N', 2.0, a, b, 0.2, copy(c))
        backend_generic()
        @test NDTensors.gemm_backend[] == :Generic
        res3 = NDTensors._gemm!('N', 'N', 2.0, a, b, 0.2, copy(c))
        @test res1 == res2
        @test res1 ≈ res3
        backend_auto()
    end

    @testset "change backends" begin
        a, b, c = [randn(5, 5) for i in 1:3]
        backend_auto()
        @test NDTensors.gemm_backend[] == :Auto
        @test NDTensors.auto_select_backend(typeof.((a, b, c))...) ==
            NDTensors.GemmBackend(:BLAS)
        res1 = NDTensors._gemm!('N', 'N', 2.0, a, b, 0.2, copy(c))
        @test_throws UndefVarError backend_octavian()
        if VERSION >= v"1.5"
            # Octavian only support Julia 1.5
            # Need to install it here instead of
            # putting it as a dependency in the Project.toml
            # since otherwise it fails for older Julia versions.
            using Octavian
            NDTensors.backend_octavian()
            @test NDTensors.gemm_backend[] == :Octavian
            res4 = NDTensors._gemm!('N', 'N', 2.0, a, b, 0.2, copy(c))
            @test res1 ≈ res4
            backend_auto()
        end
    end
end

nothing
end
