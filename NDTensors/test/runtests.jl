using SafeTestsets: @safetestset

@safetestset "NDTensors" begin
    using Test: @testset
    using NDTensors: NDTensors
    @testset "$(@__DIR__)" begin
        filenames = filter(readdir(@__DIR__)) do f
            startswith("test_")(f) && endswith(".jl")(f)
        end
        for dir in ["lib"]
            push!(filenames, joinpath(dir, "runtests.jl"))
        end
        @testset "Test $(@__DIR__)/$filename" for filename in filenames
            println("Running $(@__DIR__)/$filename")
            include(filename)
        end
    end
end

nothing
