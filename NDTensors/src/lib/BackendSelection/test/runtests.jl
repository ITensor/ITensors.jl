@eval module $(gensym())
using Test: @test, @testset
using NDTensors: NDTensors
using NDTensors.BackendSelection:
    BackendSelection, Algorithm, Backend, @Algorithm_str, @Backend_str
# TODO: This is defined for backwards compatibility,
# delete this alias once downstream packages change over
# to using `BackendSelection`.
using NDTensors.AlgorithmSelection: AlgorithmSelection
@testset "BackendSelection" begin
    # TODO: This is defined for backwards compatibility,
    # delete this alias once downstream packages change over
    # to using `BackendSelection`.
    @test AlgorithmSelection === BackendSelection
    for type in (Algorithm, Backend)
        @testset "$type" begin
            @test type("backend") isa type{:backend}
            @test type(:backend) isa type{:backend}
            backend = type("backend"; x = 2, y = 3)
            @test backend isa type{:backend}
            @test BackendSelection.parameters(backend) === (; x = 2, y = 3)
        end
    end
    # Macro syntax.
    @test Algorithm"backend"(; x = 2, y = 3) === Algorithm("backend"; x = 2, y = 3)
    @test Backend"backend"(; x = 2, y = 3) === Backend("backend"; x = 2, y = 3)
    @test isnothing(show(Algorithm("")))
end
end
