@eval module $(gensym())
using ITensors.ITensorMPS: MPS, maxlinkdim, randomMPS, siteinds
using LinearAlgebra: norm
using Test: @test, @testset
@testset "randomMPS" begin
  sites = siteinds("S=1/2", 4)
  state = j -> isodd(j) ? "↑" : "↓"
  linkdims = 2
  # Deprecated linkdims syntax
  for mps in [
    randomMPS(Float64, sites, state; linkdims),
    randomMPS(Float64, sites; linkdims),
    randomMPS(sites, state; linkdims),
    randomMPS(sites, linkdims),
    # Deprecated linkdims syntax
    randomMPS(Float64, sites, state, linkdims),
    randomMPS(Float64, sites, linkdims),
    randomMPS(sites, state, linkdims),
    randomMPS(sites, linkdims),
  ]
    @test mps isa MPS
    @test length(mps) == 4
    @test maxlinkdim(mps) == 2
    @test norm(mps) > 0
  end
end
end
