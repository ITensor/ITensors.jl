@eval module $(gensym())
using NDTensors
using Test: @test, @testset

# TBLIS.jl only builds bindings for the platforms tblis_jll ships binaries for and
# errors on load anywhere else, so the extension can only be exercised on x86_64.
if Sys.ARCH === :x86_64
    using TBLIS: TBLIS

    @testset "NDTensorsTBLISExt contract! elt=$elt" for elt in (Float32, Float64)
        T1 = randomTensor(elt, (3, 4))
        T2 = randomTensor(elt, (4, 5))
        A1, A2 = convert(Array, T1), convert(Array, T2)

        @testset "α=$α, β=$β" for (α, β) in ((one(elt), zero(elt)), (elt(2), elt(3)))
            R0 = randomTensor(elt, (3, 5))
            A0 = convert(Array, R0)

            # Dispatch on `Val(:TBLIS)` directly, so this fails rather than quietly
            # falling back to the default backend if the extension is not loaded.
            R_tblis = copy(R0)
            NDTensors.contract!(
                Val(:TBLIS), R_tblis, (1, 2), T1, (1, -1), T2, (-1, 2), α, β
            )

            R = copy(R0)
            NDTensors.contract!(R, (1, 2), T1, (1, -1), T2, (-1, 2), α, β)

            @test convert(Array, R_tblis) ≈ convert(Array, R)
            @test convert(Array, R_tblis) ≈ α * A1 * A2 + β * A0
        end
    end
end
end
