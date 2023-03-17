using NDTensors
using Test

@testset "Dense Tensors" begin
  @testset "DenseTensor basic functionality" begin
    A = Tensor((3, 4))

    for I in eachindex(A)
      @test A[I] == 0
    end

    @test A[2, 1] isa Float64
    @test dims(A[1:2, 1]) == (2,)
    @test dims(A[1:2, 2]) == (2,)
    @test dims(A[2:3, 2]) == (2,)
    @test dims(A[2, 2:4]) == (3,)
    @test dims(A[2:3, 2:4]) == (2, 3)
    @test dims(A[2:3, 2:end]) == (2, 3)
    @test dims(A[3, 2:end]) == (3,)

    randn!(A)

    for I in eachindex(A)
      @test A[I] != 0
    end

    for I in eachindex(A)
      @test A[I] != 0
    end

    @test ndims(A) == 2
    @test dims(A) == (3, 4)
    @test inds(A) == (3, 4)

    A[1, 1] = 11

    @test A[1, 1] == 11

    Aview = A[2:3, 2:3]

    @test dims(Aview) == (2, 2)
    @test A[2, 2] == Aview[1, 1]

    @test A * 2.0 == 2.0 * A

    Asim = similar(data(A), 10)
    @test eltype(Asim) == Float64
    @test length(Asim) == 10

    B = Tensor(undef, (3, 4))
    randn!(B)

    C = A + B

    for I in eachindex(C)
      @test C[I] == A[I] + B[I]
    end

    Ap = permutedims(A, (2, 1))

    for I in eachindex(A)
      @test A[I] == Ap[NDTensors.permute(I, (2, 1))]
    end

    t = Tensor(ComplexF64, (100, 100))
    randn!(t)
    @test conj(data(store(t))) == data(store(conj(t)))
    @test typeof(conj(t)) <: DenseTensor

    @test Dense(ComplexF64) == Dense{ComplexF64}()
    @test Dense(ComplexF64) == complex(Dense(Float64))

    D = Tensor(ComplexF64, (100, 100))
    @test eltype(D) == ComplexF64
    @test ndims(D) == 2
    @test dim(D) == 100^2

    E = Tensor(ComplexF64, undef, (100, 100))
    @test eltype(E) == ComplexF64
    @test ndims(E) == 2
    @test dim(E) == 100^2

    F = Tensor((100, 100))
    @test eltype(F) == Float64
    @test ndims(F) == 2
    @test dim(F) == 100^2

    G = Tensor(undef, (100, 100))
    @test eltype(G) == Float64
    @test ndims(G) == 2
    @test dim(G) == 100^2

    H = Tensor(ComplexF64, undef, (100, 100))
    @test eltype(H) == ComplexF64
    @test ndims(H) == 2
    @test dim(H) == 100^2

    I_arr = rand(10, 10, 10)
    I = Tensor(I_arr, (10, 10, 10))
    @test eltype(I) == Float64
    @test dim(I) == 1000
    @test Array(I) == I_arr

    J = Tensor((2, 2))
    K = Tensor((2, 2))
    @test Array(J * K) ≈ Array(J) * Array(K)
  end

  @testset "Random constructor" begin
    T = randomTensor((2, 2))
    @test dims(T) == (2, 2)
    @test eltype(T) == Float64
    @test T[1, 1] ≉ 0
    @test norm(T) ≉ 0

    Tc = randomTensor(ComplexF64, (2, 2))
    @test dims(Tc) == (2, 2)
    @test eltype(Tc) == ComplexF64
    @test Tc[1, 1] ≉ 0
    @test norm(Tc) ≉ 0
  end

  @testset "Complex Valued Tensors" begin
    d1, d2, d3 = 2, 3, 4
    T = randomTensor(ComplexF64, (d1, d2, d3))

    rT = real(T)
    iT = imag(T)
    cT = conj(T)

    for n1 in 1:d1, n2 in 1:d2, n3 in 1:d3
      @test rT[n1, n2, n3] ≈ real(T[n1, n2, n3])
      @test iT[n1, n2, n3] ≈ imag(T[n1, n2, n3])
      @test cT[n1, n2, n3] ≈ conj(T[n1, n2, n3])
    end
  end

  @testset "Custom inds types" begin
    struct MyInd
      dim::Int
    end
    NDTensors.dim(i::MyInd) = i.dim

    T = Tensor((MyInd(2), MyInd(3), MyInd(4)))
    @test store(T) isa Dense
    @test eltype(T) == Float64
    @test norm(T) == 0
    @test dims(T) == (2, 3, 4)
    @test ndims(T) == 3
    @test inds(T) == (MyInd(2), MyInd(3), MyInd(4))
    T[2, 1, 2] = 1.21
    @test T[2, 1, 2] == 1.21
    @test norm(T) == 1.21

    T = randomTensor(ComplexF64, (MyInd(4), MyInd(3)))
    @test store(T) isa Dense
    @test eltype(T) == ComplexF64
    @test norm(T) > 0
    @test dims(T) == (4, 3)
    @test ndims(T) == 2
    @test inds(T) == (MyInd(4), MyInd(3))

    T2 = 2 * T
    @test eltype(T2) == ComplexF64
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
      R = Tensor(ComplexF64, (2, 2, 1))
      fill!(R, NaN)
      @test any(isnan, R)
      T1 = randomTensor((2, 2, 1))
      T2 = randomTensor(ComplexF64, (1, 1))
      NDTensors.contract!(R, (1, 2, 3), T1, (1, 2, -1), T2, (-1, 1))
      @test !any(isnan, R)
      @test convert(Array, R) ≈ convert(Array, T1) * T2[1]
    end

    @testset "Permutation" begin
      R = Tensor(ComplexF64, (2, 2, 1))
      fill!(R, NaN)
      @test any(isnan, R)
      T1 = randomTensor((2, 2, 1))
      T2 = randomTensor(ComplexF64, (1, 1))
      NDTensors.contract!(R, (2, 1, 3), T1, (1, 2, -1), T2, (-1, 1))
      @test !any(isnan, R)
      @test convert(Array, R) ≈ permutedims(convert(Array, T1), (2, 1, 3)) * T2[1]
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
      using Pkg: Pkg
      Pkg.add("Octavian")
      using Octavian
      backend_octavian()
      @test NDTensors.gemm_backend[] == :Octavian
      res4 = NDTensors._gemm!('N', 'N', 2.0, a, b, 0.2, copy(c))
      @test res1 ≈ res4
      backend_auto()
    end
  end
end

nothing
