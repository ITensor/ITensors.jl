using ITensors
using Test

@testset "CP-decomposition of random 2x2x2 tensors" begin
    Ntrials = 111
    for trials in 1:Ntrials
        X = randomITensor(Index(2), Index(2), Index(2))

        # """The set of 2×2×2 tensors of rank two fills about 79% of the
        # space, while those of rank three fill 21%""" KB09

        modelrank = 3
        λ, A, iteration = cp_als(X, modelrank)

        X̂ = prod(A) * λ
        err = norm(X-X̂)
        @test err < 1e-10

        # println("Ended in $iteration iterations.")
        # println(λ)
        # for An in A
        #     println(array(An))
        # end
        # println("INPUT")
        # println(X)
        # println("PARAFAC")
        # println(X̂)
        # println("Error: $err")
    end
end
