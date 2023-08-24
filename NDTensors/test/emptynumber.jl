using NDTensors
using LinearAlgebra
using Test

const 𝟎 = NDTensors.UnspecifiedZero()

@testset "NDTensors.UnspecifiedZero" begin
  x = 2.3

  @test complex(𝟎) == 𝟎
  @test complex(NDTensors.UnspecifiedZero) == Complex{NDTensors.UnspecifiedZero}

  # Basic arithmetic
  @test 𝟎 + 𝟎 == 𝟎
  @test 𝟎 + x == x
  @test x + 𝟎 == x
  @test -𝟎 == 𝟎
  @test 𝟎 - 𝟎 == 𝟎
  @test x - 𝟎 == x
  @test 𝟎 * 𝟎 == 𝟎
  @test x * 𝟎 == 𝟎
  @test 𝟎 * x == 𝟎
  @test 𝟎 / x == 𝟎
  @test_throws DivideError() x / 𝟎 == 𝟎
  @test_throws DivideError() 𝟎 / 𝟎 == 𝟎

  @test float(𝟎) == 0.0
  @test float(𝟎) isa Float64
  @test norm(𝟎) == 0.0
  @test norm(𝟎) isa Float64
end
