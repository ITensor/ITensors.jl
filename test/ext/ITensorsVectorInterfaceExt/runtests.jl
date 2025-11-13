@eval module $(gensym())
using ITensors: ITensor, Index, dag, inds, random_itensor
using Test: @test, @testset
using VectorInterface:
    add,
    add!,
    add!!,
    inner,
    scalartype,
    scale,
    scale!,
    scale!!,
    zerovector,
    zerovector!,
    zerovector!!

const elts = (Float32, Float64, Complex{Float32}, Complex{Float64})
@testset "ITensorsVectorInterfaceExt (eltype=$elt)" for elt in elts
    i, j, k = Index.((2, 2, 2))
    a = random_itensor(elt, i, j, k)
    b = random_itensor(elt, k, i, j)
    α = randn(elt)
    β = randn(elt)
    αᶜ = randn(complex(elt))
    βᶜ = randn(complex(elt))

    # add
    @test add(a, b) ≈ a + b
    @test add(a, b, α) ≈ a + α * b
    @test add(a, b, α, β) ≈ β * a + α * b

    @test add(a, b, αᶜ) ≈ a + αᶜ * b
    @test add(a, b, αᶜ, βᶜ) ≈ βᶜ * a + αᶜ * b

    # add!
    a′ = copy(a)
    add!(a′, b)
    @test a′ ≈ a + b
    a′ = copy(a)
    add!(a′, b, α)
    @test a′ ≈ a + α * b
    a′ = copy(a)
    add!(a′, b, α, β)
    @test a′ ≈ β * a + α * b

    # add!!
    a′ = copy(a)
    add!!(a′, b)
    @test a′ ≈ a + b
    a′ = copy(a)
    add!!(a′, b, α)
    @test a′ ≈ a + α * b
    a′ = copy(a)
    add!!(a′, b, α, β)
    @test a′ ≈ β * a + α * b

    a′ = copy(a)
    a′ = add!!(a′, b, αᶜ)
    @test a′ ≈ a + αᶜ * b
    a′ = copy(a)
    a′ = add!!(a′, b, αᶜ, βᶜ)
    @test a′ ≈ βᶜ * a + αᶜ * b

    # inner
    @test inner(a, b) ≈ (dag(a) * b)[]
    @test inner(a, a) ≈ (dag(a) * a)[]

    # scalartype
    @test scalartype(a) === elt
    @test scalartype(b) === elt
    @test scalartype([a, b]) === elt
    @test scalartype([a, random_itensor(Float32, i, j)]) === elt
    @test scalartype(ITensor[]) === Bool

    # scale
    @test scale(a, α) ≈ α * a

    @test scale(a, αᶜ) ≈ αᶜ * a

    # scale!
    a′ = copy(a)
    scale!(a′, α)
    @test a′ ≈ α * a
    a′ = copy(a)
    scale!(a′, b, α)
    @test a′ ≈ α * b

    # scale!!
    a′ = copy(a)
    scale!!(a′, α)
    @test a′ ≈ α * a
    a′ = copy(a)
    scale!!(a′, b, α)
    @test a′ ≈ α * b

    a′ = copy(a)
    a′ = scale!!(a′, αᶜ)
    @test a′ ≈ αᶜ * a
    a′ = copy(a)
    a′ = scale!!(a′, b, αᶜ)
    @test a′ ≈ αᶜ * b

    # zerovector
    z = zerovector(a)
    @test iszero(z)
    @test issetequal(inds(a), inds(z))
    @test eltype(z) === eltype(a)

    z = zerovector(a, complex(elt))
    @test iszero(z)
    @test issetequal(inds(a), inds(z))
    @test eltype(z) === complex(eltype(a))

    # zerovector!
    z = copy(a)
    zerovector!(z)
    @test iszero(z)
    @test issetequal(inds(a), inds(z))
    @test eltype(z) === eltype(a)

    # zerovector!!
    z = copy(a)
    zerovector!!(z, elt)
    @test iszero(z)
    @test issetequal(inds(a), inds(z))
    @test eltype(z) === eltype(a)

    z = copy(a)
    z = zerovector!!(z, complex(elt))
    @test iszero(z)
    @test issetequal(inds(a), inds(z))
    @test eltype(z) === complex(eltype(a))
end
end
