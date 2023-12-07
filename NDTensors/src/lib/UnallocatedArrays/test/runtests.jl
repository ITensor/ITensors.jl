#@eval module $(gensym())
using FillArrays: FillArrays, AbstractFill, Fill, Zeros
using NDTensors: NDTensors
using NDTensors.UnallocatedArrays
using LinearAlgebra: norm
using Test: @test, @testset, @test_broken
## Could potentially need this
#using GPUArraysCore: @allowscalar

include(joinpath(pkgdir(NDTensors), "test", "NDTensorsTestUtils", "NDTensorsTestUtils.jl"))
using .NDTensorsTestUtils: devices_list

#@testset "Testing UnallocatedArrays" for dev in devices_list(ARGS),
 # elt in (Float64, Float32, ComplexF64, ComplexF32)

  z = Zeros{elt}((2, 3))
  Z = UnallocatedZeros(z, dev(Matrix{eltype(z)}))

  @test Z isa AbstractFill
  @test size(Z) == (2, 3)
  @test length(Z) == 6
  @test sum(Z) == 0
  @test norm(Z) == 0
  @test Z[2, 3] == 0
  @test allocate(Z) isa dev(Matrix{elt})
  Zp = set_alloctype(z, dev(Matrix{elt}))
  @test Zp == Z
  Zc = copy(Z)
  @test Zc == Z
  Zc = complex(Z)
  @test eltype(Zc) == complex(eltype(z))
  @test Zc[1, 2] == 0.0 + 0.0im

  ## Things that are still broken
  R = Zc * Zc'
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)
  @test size(R) == (2,2)
  M = rand(elt, (3,4))
  R = Zc * M
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)
  @test size(R) == (2,4)
  R = M' * Zc'
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)
  @test size(R) == (4,2)
  R = transpose(M) * transpose(Zc)
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)
  @test size(R) == (4,2)
  
  R = eltype(Zc)(2.0) .* Zc
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)
  R = Zc .* eltype(Zc)(2.0)
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)
  
  R = Zc .* Zc
  @test R isa UnallocatedZeros
  @test alloctype(R) == alloctype(Zc)

  A = UnallocatedZeros(Zeros{elt}(2), Vector{Float32})
  B = UnallocatedZeros(Zeros{elt}(2), Vector{Float32})
  C = kron(A, B)
  @test C isa UnallocatedZeros
  @test alloctype(C) == alloctype(B)
  ## TODO make other kron tests

  ## The following two tests don't work properly yet
  R = Zc + Zc
  @test_broken R isa UnallocatedZeros
  @test_broken alloctype(R) == alloctype(Zc)
  R = Zc .+ eltype(Zc)(2.0)
  @test_broken R isa UnallocatedFill
  
  #########################################
  # UnallocatedFill
  f = Fill{elt}(3.0, (2, 3, 4))
  F = UnallocatedFill{elt,ndims(f),typeof(axes(f)),Array{elt,ndims(f)}}(f)
  @test F isa AbstractFill
  @test size(F) == (2, 3, 4)
  @test length(F) == 24
  @test sum(F) ≈ 3 * 24
  @test norm(F) ≈ sqrt(3^2 * 24)
  @test F[2, 3, 1] == 3.0
  @test allocate(F) isa Array{elt,3}
  Fp = allocate(F)
  @test norm(Fp) ≈ norm(F)

  Fp = set_alloctype(f, dev(Array{elt,ndims(f)}))
  @test allocate(Fp) isa dev(Array{elt,ndims(f)})
  @test Fp == F
  Fc = copy(F)
  @test Fc == F
  Fc = allocate(complex(F))
  @test eltype(Fc) == complex(eltype(F))
  @test typeof(Fc) == alloctype(complex(F))
  ## This allocates is this correct?
  ## TODO this is broken because it doesn't call allocate
  Fc[2, 3, 4] = 4.0 + 3.0im
  @test Fc[2, 3, 4] == 4.0 + 3.0im
end
end
