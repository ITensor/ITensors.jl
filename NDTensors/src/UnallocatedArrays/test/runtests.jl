using FillArrays
using NDTensors.UnallocatedArrays
using LinearAlgebra
using Test

begin
  z = Zeros{Float64}((2, 3))
  Z = UnallocatedZeros{eltype(z),ndims(z),typeof(axes(z)),Matrix{eltype(z)}}(z)

  @test size(Z) == (2, 3)
  @test length(Z) == 6
  @test sum(Z) == 0
  @test norm(Z) == 0
  @test Z[2, 3] == 0
  @test allocate(Z) isa Matrix{eltype(z)}
  Zp = set_alloctype(z, Matrix{eltype(z)})
  @test Zp == Z
  Zc = copy(Z)
  @test Zc == Z

  # z = Zeros(())
  # Z = UnallocatedZeros{Float32, ndims(z), typeof(axes(z)), Vector{Float32}}(z)
  # @test size(Z) == ()
  # @test length(Z) == 1
  # @test Z[] == 0
  # Z = set_alloctype(z, Matrix{Float64})
  # m = allocate(Z)
  # @test length(m) == 1
  # m[] == 0

  f = Fill{Float64}(3.0, (2, 3, 4))
  F = UnallocatedFill{eltype(f),ndims(f),typeof(axes(f)),Array{eltype(f),ndims(f)}}(f)
  @test size(F) == (2, 3, 4)
  @test length(F) == 24
  @test sum(F) ≈ 3 * 24
  @test norm(F) ≈ sqrt(3^2 * 24)
  @test F[2, 3, 1] == 3.0
  @test allocate(F) isa Array{eltype(z),3}
  Fp = allocate(F)
  @test norm(Fp) ≈ norm(F)
  Fp = set_alloctype(f, Array{eltype(f),ndims(f)})
  @test Fp == F
  Fc = copy(F)
  @test Fc == F
end
