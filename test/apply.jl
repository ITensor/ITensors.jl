using ITensors
using Test

@testset "apply" begin

  @testset "ITensors.minimal_swap_range" begin
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 2], [1, 2]) == 1:2
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 2], [3, 4]) == 1:2
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 3], [5, 6]) == 2:3
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 4], [5, 6]) == 3:4
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [3, 6], [1, 4]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 5], [5, 6]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 6], [5, 6]) == 5:6
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 6], [4, 5, 6]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 3, 5], [7, 8]) == 3:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 5], [2, 3]) == 2:3
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 5], [2, 4]) == 2:3
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 3, 5], [2, 4]) == 2:4
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 3, 5], [5, 8]) == 3:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [3, 4], [1, 2]) == 3:4
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 6], [1, 2]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 6], [1, 3]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 6], [1, 4]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [2, 6], [1, 6]) == 2:3
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [2, 6], [1, 7]) == 2:3
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [2, 6], [1, 5]) == 5:6
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 5], [2, 6]) == 1:2
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 3, 5], [2, 6]) == 1:3
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [2, 5, 6], [1, 5]) == 4:6
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [1, 5], [2, 6]) == 1:2
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 8], [3, 8]) == 4:5
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 6, 8], [3, 5]) == 4:6
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 6, 8], [3, 4]) == 4:6
    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [4, 6, 8], [3, 8]) == 4:6
  end

  @testset "apply swap" begin
    N = 6
    s = siteinds("Qubit", N)

    Xop = [("X", n) for n in 1:N]
    CXop = [("CX", i, j) for i in 1:N, j in 1:N]
    CCXop = [("CCX", i, j, k) for i in 1:N, j in 1:N, k in 1:N]

    X = op.(Xop, (s,))
    CX = op.(CXop, (s,))
    CCX = op.(CCXop, (s,))

    ψ0 = MPS(s, "0")
    orthogonalize!(ψ0, 1)
    @test ortho_lims(ψ0) == 1:1

    move_sites_back = false
    kwargs = (; move_sites_back=move_sites_back)

    nswaps = Int[]
    ψ = apply(X[1], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 1:1
    @test nswaps == [0]

    nswaps = Int[]
    ψ = apply(X[3], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 3:3
    @test nswaps == [0]

    nswaps = Int[]
    ψ = apply(CX[1, 2], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 2:2
    @test nswaps == [0]

    nswaps = Int[]
    ψ = apply(CX[2, 3], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 3:3
    @test nswaps == [0]

    nswaps = Int[]
    ψ = apply(CX[3, 4], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 4:4
    @test nswaps == [0]

    nswaps = Int[]
    ψ = apply([CX[3, 4], X[4]], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 4:4
    @test nswaps == [0, 0]

    nswaps = Int[]
    ψ = apply(CX[1, 3], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 3:3
    @test nswaps == [1]

    nswaps = Int[]
    ψ = apply(CX[1, 4], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 4:4
    @test nswaps == [2]

    nswaps = Int[]
    ψ = apply([CCX[2, 3, 4], X[2]], ψ0; (nswaps!)=nswaps, kwargs...)
    @test ortho_lims(ψ) == 2:2
    @test nswaps == [0, 0]

    nswaps = Int[]
    ψ = apply([CX[3, 5], CX[1, 3]], ψ0; (nswaps!)=nswaps, kwargs...)

    nswaps = Int[]
    ψ = apply(CX[1, 4], ψ0; next_gate=CX[2, 3], (nswaps!)=nswaps, kwargs...)
    @test nswaps == [2]

    nswaps = Int[]
    ψ = apply([CX[3, 6], CX[1, 3]], ψ0; (nswaps!)=nswaps, kwargs...)
    @test nswaps == [2, 1]

    @test ITensors.consecutive_range(ITensors.SwapMinimal(), [3, 6], [1, 4]) == 4:5
    nswaps = Int[]
    ψ = apply([CX[3, 6], CX[1, 4]], ψ0; (nswaps!)=nswaps, kwargs...)
    @test nswaps == [2, 1]

    # XXX broken
    nswaps = Int[]
    ψ = apply([CX[1, 3], CX[3, 5]], ψ0; next_gate=CX[3, 5], (nswaps!)=nswaps, kwargs...)
    @test nswaps == [1, 1]

    # XXX broken
    nswaps = Int[]
    ψ = apply([CX[1, 3], CX[3, 5]], ψ0; next_gate=CX[3, 5], (nswaps!)=nswaps, kwargs...)
    @test nswaps == [1, 1]

  end

  @testset "Debugging" begin
    N = 4
    s = siteinds("Qubit", N)
    X = [op("X", s, n) for n in 1:N]
    ψ = MPS(s, "0")
    @test prod(product(X[1], ψ)) ≈ prod(MPS(s, n -> n == 1 ? "1" : "0"))
  end
end

