using ITensors
using Test

ITensors.Strided.disable_threads()
ITensors.BLAS.set_num_threads(1)
ITensors.disable_threaded_blocksparse()

@testset "ITensors tests" begin
    # Make a copy in case a test modifies it.
    test_args = copy(ARGS)
    println("Passed arguments ARGS = $(test_args) to tests.")
    if isempty(test_args) || "all" in test_args || "base" in test_args
        println(
            """\nArguments ARGS = $(test_args) are empty, or contain `"all"` or `"base"`. Running base (non-MPS/MPO) ITensors tests.""",
        )
        dirs = [
            "lib/LazyApply",
            "lib/Ops",
            "base",
            "threading",
            "ext/ITensorsChainRulesCoreExt",
            "ext/ITensorsTensorOperationsExt",
            "ext/ITensorsVectorInterfaceExt",
            "ext/NDTensorsMappedArraysExt",
        ]
        @time for dir in dirs
            println("\nTest $(@__DIR__)/$(dir)")
            @time include(joinpath(@__DIR__, dir, "runtests.jl"))
            if ARGS ≠ test_args
                # Fix ARGS in case a test modifies it.
                append!(empty!(ARGS), test_args)
            end
        end
    end
end
