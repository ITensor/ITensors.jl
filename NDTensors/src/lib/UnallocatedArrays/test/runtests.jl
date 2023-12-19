@eval module $(gensym())
using FillArrays: FillArrays, AbstractFill, Fill, Zeros
using NDTensors: NDTensors
using NDTensors.UnallocatedArrays
using LinearAlgebra: norm
using Test: @test, @testset, @test_broken

include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list

@testset "Testing UnallocatedArrays" for dev in devices_list(ARGS),
  elt in (Float64, Float32, ComplexF64, ComplexF32)

  @testset "Basic funcitonality" begin
    z = Zeros{elt}((2, 3))
    Z = UnallocatedZeros(z, dev(Matrix{eltype(z)}))

    @test Z isa AbstractFill
    @test size(Z) == (2, 3)
    @test length(Z) == 6
    @test iszero(sum(Z))
    @test iszero(norm(Z))
    @test iszero(Z[2, 3])
    @test allocate(Z) isa dev(Matrix{elt})
    Zp = set_alloctype(z, dev(Matrix{elt}))
    @test Zp == Z
    Zc = copy(Z)
    @test Zc == Z
    Zc = complex(Z)
    @test eltype(Zc) == complex(eltype(z))
    @test iszero(Zc[1, 2])

    Zs = similar(Z)
    @test Zs isa alloctype(Z)

    #########################################
    # UnallocatedFill
    f = Fill{elt}(3, (2, 3, 4))
    F = UnallocatedFill(f, Array{elt,ndims(f)})
    @test F isa AbstractFill
    @test size(F) == (2, 3, 4)
    @test length(F) == 24
    @test sum(F) ≈ elt(3) * 24
    @test norm(F) ≈ sqrt(elt(3)^2 * 24)
    @test F[2, 3, 1] == elt(3)
    @test allocate(F) isa Array{elt,3}
    Fp = allocate(F)
    @test norm(Fp) ≈ norm(F)
    Fs = similar(F)
    @test Fs isa alloctype(F)
    @test Fs[1, 1, 1] != elt(3)

    Fp = set_alloctype(f, dev(Array{elt,ndims(f)}))
    @test allocate(Fp) isa dev(Array{elt,ndims(f)})
    @test Fp == F
    Fc = copy(F)
    @test Fc == F
    Fc = allocate(complex(F))
    @test eltype(Fc) == complex(eltype(F))
    @test typeof(Fc) == alloctype(complex(F))
    Fc[2, 3, 4] = elt(0)
    @test iszero(Fc[2, 3, 4])
  end

  @testset "Multiplication" begin
    z = Zeros{elt}((2, 3))
    Z = UnallocatedZeros(z, dev(Matrix{eltype(z)}))

    R = Z * Z'
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)
    @test size(R) == (2, 2)
    M = rand(elt, (3, 4))
    R = Z * M
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)
    @test size(R) == (2, 4)
    R = M' * Z'
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)
    @test size(R) == (4, 2)
    R = transpose(M) * transpose(Z)
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)
    @test size(R) == (4, 2)

    ###################################
    ## UnallocatedFill
    f = Fill{elt}(3, (2, 12))
    F = UnallocatedFill(f, dev(Matrix{elt}))
    p = Fill{elt}(4, (12, 5))
    P = UnallocatedFill(p, dev(Array{elt,ndims(p)}))
    R = F * P
    @test F isa UnallocatedFill
    @test R[1, 1] == elt(144)
    @test alloctype(R) == alloctype(F)
    @test size(R) == (2, 5)

    R = F * F'
    @test R isa UnallocatedFill
    @test R[1, 2] == elt(108)
    @test alloctype(R) == alloctype(F)
    @test size(R) == (2, 2)

    R = transpose(F) * F
    @test R isa UnallocatedFill
    @test R[12, 3] == elt(18)
    @test alloctype(R) == alloctype(F)
    @test size(R) == (12, 12)

    R = transpose(Z) * F
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)
    @test size(R) == (3, 12)
  end

  @testset "Broadcast" begin
    z = Zeros{elt}((2, 3))
    Z = UnallocatedZeros(z, dev(Matrix{elt}))
    R = elt(2) .* Z
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)
    R = Z .* elt(2)
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)

    R = Z .* Z
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)

    Z = UnallocatedZeros(Zeros{elt}((2, 3)), dev(Matrix{elt}))
    R = Z + Z
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(Z)

    R = Z .+ elt(2)
    @test R isa UnallocatedFill
    @test alloctype(R) == alloctype(Z)
    ########################
    # UnallocatedFill
    f = Fill(elt(3), (2, 3, 4))
    F = UnallocatedFill(f, Array{elt,ndims(f)})
    F2 = F .* 2
    @test F2 isa UnallocatedFill
    @test F2[1, 1, 1] == elt(6)
    @test alloctype(F2) == alloctype(F)

    F2 = F2 .+ elt(2)
    @test F2 isa UnallocatedFill
    @test F2[1, 1, 1] == elt(8)
    @test alloctype(F2) == alloctype(F)

    F = UnallocatedFill(Fill(elt(2), (2, 3)), dev(Matrix{elt}))
    R = Z + F
    @test R isa UnallocatedFill
    @test alloctype(R) == alloctype(Z)

    R = F + Z
    @test R isa UnallocatedFill
    @test alloctype(R) == alloctype(Z)

    F = UnallocatedFill(Fill(elt(3), (2, 12)), dev(Matrix{elt}))
    R = F .* F
    @test R isa UnallocatedFill
    @test R[2, 9] == elt(9)
    @test alloctype(R) == alloctype(F)
    @test size(R) == (2, 12)

    P = UnallocatedFill(Fill(elt(4), (2, 3)), dev(Matrix{elt}))
    R = Z .* P
    @test R isa UnallocatedZeros
    @test alloctype(R) == alloctype(P)
    @test size(R) == (2, 3)

    F = UnallocatedFill(Fill(elt(2), (2, 3)), dev(Matrix{elt}))
    R = F + F
    @test R isa UnallocatedFill
  end

  ## TODO make other kron tests
  @testset "Kron" begin
    A = UnallocatedZeros(Zeros{elt}(2), dev(Vector{elt}))
    B = UnallocatedZeros(Zeros{elt}(2), dev(Vector{elt}))
    C = kron(A, B)
    @test C isa UnallocatedZeros
    @test alloctype(C) == alloctype(B)

    B = UnallocatedFill(Fill(elt(2), (2)), dev(Vector{elt}))
    C = kron(A, B)
    @test C isa UnallocatedZeros
    @test alloctype(C) == alloctype(B)

    C = kron(B, A)
    @test C isa UnallocatedZeros
    @test alloctype(C) == alloctype(B)

    A = UnallocatedFill(Fill(elt(3), (2)), dev(Vector{elt}))
    C = kron(A, B)
    @test C isa UnallocatedFill
    @test alloctype(C) == alloctype(B)
    @test C[1] == elt(6)
  end
end
end
