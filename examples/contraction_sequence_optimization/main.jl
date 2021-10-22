using ITensors

import ITensors: optimal_contraction_sequence

# This is for daggering and Empty ITensor
using ITensors.NDTensors
Base.conj(s::Empty; kwargs...) = s

# TODO: use ITensors.contraction_cost
contraction_cost(A::Vector{<:ITensor}, sequence) = _contraction_cost(A, sequence, 0)
function _contraction_cost(A::Vector{<:ITensor}, sequence, previouscost)
  inds1, cost1 = _contraction_cost(A, sequence[1], previouscost)
  inds2, cost2 = _contraction_cost(A, sequence[2], previouscost)
  return _contraction_cost(inds1, inds2, previouscost + cost1 + cost2)
end
function _contraction_cost(As::Vector{<:ITensor}, sequence::Int, previouscost)
  return inds(As[sequence]), previouscost
end
function _contraction_cost(A1::IndexSet, A2::IndexSet, previouscost)
  return noncommoninds(A1, A2), previouscost + dim(unioninds(A1, A2))
end

# "Contract" a set of labels according to a sequence
function contract_labels(network_labels::Vector{Symbol}, sequence)
  branch1 = contract_labels(network_labels, sequence[1])
  branch2 = contract_labels(network_labels, sequence[2])
  return contract_labels(branch1, branch2)
end
contract_labels(network_labels::Vector{Symbol}, n::Int) = network_labels[n]
contract_labels(label1::Symbol, label2::Symbol) = (label1, label2)
contract_labels(label1::Symbol, label2::Tuple) = (label1, label2)
contract_labels(label1::Tuple, label2::Tuple) = (label1, label2)

# Examples from Section III of https://arxiv.org/abs/1304.6112

english_alphabet = Char.(97:(97 + 25))
# Include Π (Char(928))
# Include ϕ (Char(981))
# Include ϵ (Char(1013))
# Don't include \delta Char(948) since it conflicts with ITensors.δ
# Don't include \pi Char(960) since it conflicts with MathConstants.π
greek_alphabet = push!(vcat(Char.(945:947), Char.(949:959), Char.(961:969)), 928, 981, 1013)
greek_alphabet_uppercase = Char.(913:(913 + 24))
indlabels = vcat(english_alphabet, greek_alphabet, greek_alphabet_uppercase)

# Add labels with overbars as well
# \overbar Char(773)
indlabels = vcat(indlabels, indlabels .* Char(773))

function define_inds(dχ, indlabels)
  for i in indlabels
    @eval begin
      $(Symbol(i)) = Index($dχ, string($i))
    end
  end
end

# Fig. 6(a)
# Optimal sequence: Any[6, Any[Any[1, 2], Any[5, Any[3, 4]]]]
# Optimal cost: 10χ^3+ 16χ^2
function tebd(dχ, ds=2; indlabels=indlabels)
  define_inds(dχ, indlabels)

  cₛ = Index(ds, "c")
  fₛ = Index(ds, "f")
  iₛ = Index(ds, "i")
  jₛ = Index(ds, "j")

  λ¹ = randomITensor(a, b)
  Γ¹ = randomITensor(b, d, cₛ)
  λ² = randomITensor(d, e)
  Γ² = randomITensor(e, g, fₛ)
  λ³ = randomITensor(g, h)
  U = randomITensor(cₛ, fₛ, iₛ, jₛ)
  network = [λ¹, Γ¹, λ², Γ², λ³, U]
  sequence = @time optimal_contraction_sequence(network)

  # Analyze the sequence
  network_labels = [:λ¹, :Γ¹, :λ², :Γ², :λ³, :U]
  sequence_labels = @show contract_labels(network_labels, sequence)
  sequence_inds, sequence_cost = contraction_cost(network, sequence)
  @show hassameinds(sequence_inds, noncommoninds(network...))
  @show sequence_cost
  @show 10dχ^3 + 16dχ^2
  return (sequence=sequence, sequence_labels=sequence_labels, sequence_cost=sequence_cost)
end

# Fig. 6(b)
# Optimal sequence:  Any[3, Any[5, Any[2, Any[1, 4]]]]
# Optimal cost: 4χ^6
function ttn_1d_3_to_1(dχ; indlabels=indlabels)
  define_inds(dχ, indlabels)

  W² = emptyITensor(c, d, e, f)
  H = emptyITensor(a, b, k, c)
  W¹ᴴ = emptyITensor(i, j, a, h)
  W²ᴴ = replaceinds(dag(W²), (c, f) => (b, g))
  Ρ = emptyITensor(l, f, h, g)
  network = [W², H, W¹ᴴ, W²ᴴ, Ρ]
  sequence = @time optimal_contraction_sequence(network)

  # Analyze the sequence
  network_labels = [:W², :H, :W¹ᴴ, :W²ᴴ, :Ρ]
  sequence_labels = @show contract_labels(network_labels, sequence)
  sequence_inds, sequence_cost = contraction_cost(network, sequence)
  @show hassameinds(sequence_inds, noncommoninds(network...))
  @show sequence_cost
  @show 4dχ^6
  return (sequence=sequence, sequence_labels=sequence_labels, sequence_cost=sequence_cost)
end

# Fig. 7(a)
# Optimal sequence:  Any[5, Any[Any[4, Any[1, 6]], Any[Any[2, 7], Any[9, Any[3, 8]]]]]
# Optimal cost: 4χ^12+ 4χ^10
# WARNING: this takes a while to run
function ttn_2d_9_to_1(dχ; indlabels=indlabels)
  define_inds(dχ, indlabels)
  W² = emptyITensor(ϵ, ζ, η, θ, ι, κ, j, λ, μ, c)
  W³ = emptyITensor(o, p, h, q, r, s, t, u, v, a)
  W⁴ = emptyITensor(i, w, x, y, z, α, β, γ, Δ, b)
  H = emptyITensor(m, n, k, l, χ, j, h, i)
  W¹ᴴ = dag(emptyITensor(ν, ξ, Π, ρ, σ, τ, υ, ϕ, m, f))
  W²ᴴ = dag(replaceinds(W², (j, c) => (n, g)))
  W³ᴴ = dag(replaceinds(W³, (h, a) => (k, d)))
  W⁴ᴴ = dag(replaceinds(W⁴, (i, b) => (l, e)))
  Ρ = emptyITensor(ψ, c, a, b, f, g, d, e)
  network = [W², W³, W⁴, H, W¹ᴴ, W²ᴴ, W³ᴴ, W⁴ᴴ, Ρ]
  sequence = @time optimal_contraction_sequence(network)

  # Analyze the sequence
  network_labels = [:W², :W³, :W⁴, :H, :W¹ᴴ, :W²ᴴ, :W³ᴴ, :W⁴ᴴ, :Ρ]
  sequence_labels = @show contract_labels(network_labels, sequence)
  sequence_inds, sequence_cost = contraction_cost(network, sequence)
  @show hassameinds(sequence_inds, noncommoninds(network...))
  @show sequence_cost
  @show 4dχ^12 + 4dχ^10
  return (sequence=sequence, sequence_labels=sequence_labels, sequence_cost=sequence_cost)
end

# Fig. 7(b)
#
# TODO
#W(2) αδεζηθ
#W(3)  βικλμν
#W(4)  γξπρστ
#V(1)   ̄πιωabc
#V(2)   ̄ξευφχψ
#V(3)  θξdefg
#V(4)  μπhijk
#U     ψbehlmno
#h      ̄νφωlpqrs
#U∗    smnotuvw
#V(1)∗ raucxy
#V(2)∗ υqχtz ̄α
#V(3)∗ dvfg ̄β ̄γ
#V(4)∗ wijk ̄δ ̄ε
#W(1)∗  ̄λ ̄μpzx ̄ζ
#W(2)∗ δ ̄αζη ̄β ̄η
#W(3)∗ yκλ ̄δν ̄θ
#W(4)∗  ̄γ ̄ερστ ̄ι
#ρ      ̄ζ ̄η ̄θ ̄ι ̄καβγ

# Fig. 7(c)
# 4χ^26 + 2χ^25 + 2χ^23+ 3χ^22 + 3χ^20 + χ^16 + χ^14 + χ^13 + χ^12 + 4χ^8 + 4χ^7
function mera_2d_4_to_1(dχ; indlabels=indlabels)
  define_inds(dχ, indlabels)
  W² = emptyITensor(σ̅, d̅, e̅, ξ, σ)
  W³ = emptyITensor(τ̅, f̅, g̅, τ, j̅)
  W⁴ = emptyITensor(υ̅, k̅, Π, m̅, ω)
  W⁵ = emptyITensor(φ̅, ρ, υ, a, f)
  W⁶ = emptyITensor(χ̅, φ, l̅, g, n̅)
  W⁷ = emptyITensor(ψ̅, o̅, b, q̅, r̅)
  W⁸ = emptyITensor(ω̅, c, h, s̅, t̅)
  W⁹ = emptyITensor(a̅, i, p̅, u̅, v̅)
  U¹ = emptyITensor(ν, ξ, Π, ρ, κ̅, α, γ, λ̅)
  U² = emptyITensor(σ, τ, υ, φ, β, χ, μ̅, ψ)
  U³ = emptyITensor(ω, a, b, c, ν̅, ξ̅, d, e)
  U⁴ = emptyITensor(f, g, h, i, Π̅, j, k, l)
  H = emptyITensor(κ̅, α, β, γ, λ̅, μ̅, ν̅, ξ̅, Π̅, Δ, ε, ζ, η, θ, ι, κ, λ, μ)
  U¹ᴴ = dag(emptyITensor(Δ, ε, η, θ, m, n, o, p))
  U²ᴴ = dag(emptyITensor(ζ, χ, ι, ψ, s, t, u, v))
  U³ᴴ = dag(emptyITensor(κ, λ, d, e, y, z, α̅, β̅))
  U⁴ᴴ = dag(emptyITensor(μ, j, k, l, ζ̅, η̅, θ̅, ι̅))
  W¹ᴴ = dag(emptyITensor(b̅, c̅, h̅, m, w̅))
  W²ᴴ = dag(emptyITensor(d̅, e̅, n, s, x̅))
  W³ᴴ = dag(emptyITensor(f̅, g̅, t, j̅, q))
  W⁴ᴴ = dag(emptyITensor(k̅, o, m̅, y, r))
  W⁵ᴴ = dag(emptyITensor(p, u, z, ζ̅, w))
  W⁶ᴴ = dag(emptyITensor(v, l̅, η̅, n̅, x))
  W⁷ᴴ = dag(emptyITensor(o̅, α̅, q̅, r̅, γ̅))
  W⁸ᴴ = dag(emptyITensor(β̅, θ̅, s̅, t̅, Δ̅))
  W⁹ᴴ = dag(emptyITensor(ι̅, p̅, u̅, v̅, ε̅))
  Ρ = emptyITensor(w̅, x̅, q, r, w, x, γ̅, Δ̅, ε̅, ρ̅, σ̅, τ̅, υ̅, φ̅, χ̅, ψ̅, ω̅, a̅)
  network = [
    W²,
    W³,
    W⁴,
    W⁵,
    W⁶,
    W⁷,
    W⁸,
    W⁹,
    U¹,
    U²,
    U³,
    U⁴,
    H,
    U¹ᴴ,
    U²ᴴ,
    U³ᴴ,
    U⁴ᴴ,
    W¹ᴴ,
    W²ᴴ,
    W³ᴴ,
    W⁴ᴴ,
    W⁵ᴴ,
    W⁶ᴴ,
    W⁷ᴴ,
    W⁸ᴴ,
    W⁹ᴴ,
    Ρ,
  ]
  sequence = @time optimal_contraction_sequence(network)

  # Analyze the sequence
  network_labels = [
    :W²,
    :W³,
    :W⁴,
    :W⁵,
    :W⁶,
    :W⁷,
    :W⁸,
    :W⁹,
    :U¹,
    :U²,
    :U³,
    :U⁴,
    :H,
    :U¹ᴴ,
    :U²ᴴ,
    :U³ᴴ,
    :U⁴ᴴ,
    :W¹ᴴ,
    :W²ᴴ,
    :W³ᴴ,
    :W⁴ᴴ,
    :W⁵ᴴ,
    :W⁶ᴴ,
    :W⁷ᴴ,
    :W⁸ᴴ,
    :W⁹ᴴ,
    :Ρ,
  ]
  sequence_labels = @show contract_labels(network_labels, sequence)
  sequence_inds, sequence_cost = contraction_cost(network, sequence)
  @show hassameinds(sequence_inds, noncommoninds(network...))
  @show sequence_cost
  @show 4dχ^26 +
    2dχ^25 +
    2dχ^23 +
    3dχ^22 +
    3dχ^20 +
    dχ^16 +
    dχ^14 +
    dχ^13 +
    dχ^12 +
    4dχ^8 +
    4dχ^7
  return (sequence=sequence, sequence_labels=sequence_labels, sequence_cost=sequence_cost)
end
