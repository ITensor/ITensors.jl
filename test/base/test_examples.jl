@eval module $(gensym())
using ITensors: ITensors
using Suppressor: @capture_out
using Test: @test_nowarn, @testset
@testset "Example Codes" begin
    @testset "Basic Ops $filename" for filename in ["basic_ops.jl", "qn_itensors.jl"]
        @test_nowarn begin
            @capture_out begin
                include(joinpath(pkgdir(ITensors), "examples", "basic_ops", filename))
            end
        end
    end
end
end
