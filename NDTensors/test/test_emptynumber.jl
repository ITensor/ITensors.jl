@eval module $(gensym())
using LinearAlgebra: norm
using NDTensors: EmptyNumber
using Test: @testset, @test, @test_throws

const 𝟎 = EmptyNumber()

@testset "NDTensors.EmptyNumber" begin
    x = 2.3

    @test complex(𝟎) == 𝟎
    @test complex(EmptyNumber) == Complex{EmptyNumber}

    # Promotion
    for T in (Bool, Float32, Float64, Complex{Float32}, Complex{Float64})
        @test promote_type(EmptyNumber, T) === T
        @test promote_type(T, EmptyNumber) === T
    end

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
end
