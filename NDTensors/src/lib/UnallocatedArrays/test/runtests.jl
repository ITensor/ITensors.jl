@eval module $(gensym())
using FillArrays: FillArrays, AbstractFill, Fill, Zeros
using NDTensors: NDTensors
using NDTensors.UnallocatedArrays:
  UnallocatedFill, UnallocatedZeros, allocate, alloctype, set_alloctype
using LinearAlgebra: norm
using Test: @test, @test_broken, @testset

include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list

@testset "Testing UnallocatedArrays on $dev with eltype $elt" for dev in devices_list(ARGS),
  elt in (Float64, Float32, ComplexF64, ComplexF32)

  @testset "Basic funcitonality" begin
    z = Zeros{elt}((2, 3))
    Z = UnallocatedZeros(z, dev(Matrix{elt}))

    @test Z isa AbstractFill
    @test size(Z) == (2, 3)
    @test length(Z) == 6
    @test iszero(sum(Z))
    @test iszero(norm(Z))
    @test iszero(Z[2, 3])
    @test allocate(Z) isa dev(Matrix{elt})
    Zp = UnallocatedZeros{elt}(Zeros(2, 3), dev(Matrix{elt}))
    @test Zp == Z
    Zp = set_alloctype(z, dev(Matrix{elt}))
    @test Zp == Z
    Zc = copy(Z)
    @test Zc == Z
    Zc = complex(Z)
    @test eltype(Zc) == complex(eltype(z))
    @test iszero(Zc[1, 2])
    @test Zc isa UnallocatedZeros
    @test alloctype(Zc) == alloctype(Z)

    Zs = similar(Z)
    @test Zs isa alloctype(Z)

    Z = UnallocatedZeros(z, dev(Array))
    Za = allocate(Z)
    @test Za isa dev(Array{elt,2})
    @test Za[1, 3] == zero(elt)

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
    Fp = UnallocatedFill{elt}(Fill(3, (2, 3, 4)), Array{elt,ndims(f)})
    @test Fp == F
    Fp = allocate(F)
    @test norm(Fp) ≈ norm(F)
    Fs = similar(F)
    @test Fs isa alloctype(F)
    @test length(Fs) == 2 * 3 * 4
    Fs[1, 1, 1] = elt(10)
    @test Fs[1, 1, 1] == elt(10)

    Fp = set_alloctype(f, dev(Array{elt,ndims(f)}))
    @test allocate(Fp) isa dev(Array{elt,ndims(f)})
    @test Fp == F
    Fc = copy(F)
    @test Fc == F
    Fc = allocate(complex(F))
    @test eltype(Fc) == complex(eltype(F))
    ## Here we no longer require the eltype of the alloctype to
    ## Be the same as the eltype of the `UnallocatedArray`. It will be
    ## replaced when the array is allocated
    # @test_broken typeof(Fc) == alloctype(complex(F))
    Fc[2, 3, 4] = elt(0)
    @test iszero(Fc[2, 3, 4])

    F = UnallocatedFill(f, dev(Array))
    Fa = allocate(F)
    @test Fa isa dev(Array{elt,3})
    @test Fa[2, 1, 4] == elt(3)

    F = UnallocatedFill(f, dev(Vector))
    Fa = allocate(F)
    @test ndims(Fa) == 3
    @test Fa isa dev(Array)
  end

  @testset "Multiplication" begin
    z = Zeros{elt}((2, 3))
    Z = UnallocatedZeros(z, dev(Matrix{elt}))

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

    R = (x -> x .+ 1).(Z)
    @test R isa UnallocatedFill
    @test alloctype(R) == alloctype(Z)
    @test R[1, 1] == elt(1)

    Z .*= 1.0
    @test Z isa UnallocatedZeros
    @test alloctype(R) == dev(Matrix{elt})
    @test Z[2, 1] == zero(elt)
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
    @test R[1, 3] == elt(4)

    R = (x -> x .+ 1).(F)
    @test R isa UnallocatedFill
    @test R[2, 1] == elt(3)
    @test alloctype(R) == alloctype(F)
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

using FillArrays: Fill, Zeros
using NDTensors.UnallocatedArrays: UnallocatedFill, UnallocatedZeros
using NDTensors.TypeParameterAccessors:
  Position, default_type_parameter, nparameters, set_type_parameter, type_parameter
using Test: @test, @testset

@testset "SetParameters" begin
  @testset "Tetsing $typ" for (typ) in (:Fill, :Zeros)
    @eval begin
      t1 = default_type_parameter($typ, Position{1}())
      t2 = default_type_parameter($typ, Position{2}())
      t3 = default_type_parameter($typ, Position{3}())
      t4 = Any
      ft1 = $typ{t1}
      ft2 = $typ{t1,t2}
      ft3 = $typ{t1,t2,t3}

      ## check 1 parameter specified
      ftn1 = set_type_parameter(ft1, Position{1}(), t4)
      ftn2 = set_type_parameter(ft1, Position{2}(), t4)
      ftn3 = set_type_parameter(ft1, Position{3}(), t4)
      @test ftn1 == $typ{t4}
      @test ftn2 == $typ{t1,t4}
      @test ftn3 == $typ{t1,<:Any,t4}

      ## check 2 parameters specified
      ftn1 = set_type_parameter(ft2, Position{1}(), t4)
      ftn2 = set_type_parameter(ft2, Position{2}(), t4)
      ftn3 = set_type_parameter(ft2, Position{3}(), t4)
      @test ftn1 == $typ{t4,t2}
      @test ftn2 == $typ{t1,t4}
      @test ftn3 == $typ{t1,t2,t4}

      ## check 3 parameters specified
      ftn1 = set_type_parameter(ft3, Position{1}(), t4)
      ftn2 = set_type_parameter(ft3, Position{2}(), t4)
      ftn3 = set_type_parameter(ft3, Position{3}(), t4)
      @test ftn1 == $typ{t4,t2,t3}
      @test ftn2 == $typ{t1,t4,t3}
      @test ftn3 == $typ{t1,t2,t4}

      @test type_parameter(ft3, Position{1}()) == t1
      @test type_parameter(ft3, Position{2}()) == t2
      @test type_parameter(ft3, Position{3}()) == t3

      @test nparameters(ft3) == 3
    end
  end

  @testset "Tetsing $typ" for (typ) in (:UnallocatedFill, :UnallocatedZeros)
    @eval begin
      t1 = default_type_parameter($typ, Position{1}())
      t2 = default_type_parameter($typ, Position{2}())
      t3 = default_type_parameter($typ, Position{3}())
      t4 = default_type_parameter($typ, Position{4}())
      t5 = Any
      ft = $typ{t1,t2,t3,t4}

      ## check 4 parameters specified
      ftn1 = set_type_parameter(ft, Position{1}(), t5)
      ftn2 = set_type_parameter(ft, Position{2}(), t5)
      ftn3 = set_type_parameter(ft, Position{3}(), t5)
      ftn4 = set_type_parameter(ft, Position{4}(), t5)
      @test ftn1 == $typ{t5,t2,t3,t4}
      @test ftn2 == $typ{t1,t5,t3,t4}
      @test ftn3 == $typ{t1,t2,t5,t4}
      @test ftn4 == $typ{t1,t2,t3,t5}

      @test type_parameter(ft, Position{1}()) == t1
      @test type_parameter(ft, Position{2}()) == t2
      @test type_parameter(ft, Position{3}()) == t3
      @test type_parameter(ft, Position{4}()) == t4

      @test nparameters(ft) == 4
    end
  end
end
