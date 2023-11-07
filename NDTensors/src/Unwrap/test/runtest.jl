using Test
using NDTensors.Unwrap
using LinearAlgebra

## Still working on this
@testset "Testing Unwrap" begin
  v = Vector{Float64}(undef, 10)
  vt = transpose(v)
  va = v'

  E = expose(v)
  Et = expose(vt)
  Ea = expose(va)
  @test typeof(E) == Exposed{Vector{Float64},Vector{Float64}}
  @test typeof(Et) ==
    Exposed{Vector{Float64},LinearAlgebra.Transpose{Float64,Vector{Float64}}}
  @test typeof(Ea) ==
    Exposed{Vector{Float64},LinearAlgebra.Adjoint{Float64,Vector{Float64}}}

  @test parent(E) == v
  @test parent(Et) == v
  @test parent(Ea) == v
  @test transpose(E) == vt
  @test cpu(E) == v
  @test cpu(Et) == vt

  m = reshape(v, (5, 2))
  mt = transpose(m)
  ma = m'
  E = expose(m)
  Et = expose(mt)
  Ea = expose(ma)

  @test typeof(E) == Exposed{Matrix{Float64},Matrix{Float64}}
  @test typeof(Et) ==
    Exposed{Matrix{Float64},LinearAlgebra.Transpose{Float64,Matrix{Float64}}}
  @test typeof(Ea) ==
    Exposed{Matrix{Float64},LinearAlgebra.Adjoint{Float64,Matrix{Float64}}}

  o = Vector{Float32}(undef, 1)
  expose(o)[] = 2
  expose(o)[] == 2

  @which expose(m)[1, 1]
end
