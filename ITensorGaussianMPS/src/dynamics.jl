
function _compute_correlator(N::Int, f::Function; kwargs...)
  sites = get(kwargs, :sites, 1:N)
  site_range = (sites isa AbstractRange) ? sites : collect(sites)
  Ns = length(site_range)

  C = zeros(ComplexF64, Ns, Ns)
  for (ni, i) in enumerate(site_range), (nj, j) in enumerate(site_range)
    C[ni, nj] += f(i, j)
  end

  if sites isa Number
    return C[1, 1]
  end
  return C
end

"""
Compute the retarded single-particle Green function
G_R(t)_ij = -i θ(t) <ϕ|{cᵢ(t), c†ⱼ(0)}|ϕ>
where ϕ is a Slater determinant
"""
function G_R(t::Number, ϕ, ϵ::Vector{Float64}; kwargs...)
  N = length(ϵ)
  @assert size(ϕ) == (N, N)
  function compute_GR(i, j)
    gr = 0.0im
    for n in 1:N
      gr += -im * ϕ[i, n] * conj(ϕ[j, n]) * exp(-im * ϵ[n] * t)
    end
    return gr
  end
  return _compute_correlator(N, compute_GR; kwargs...)
end

"""
Compute the greater single-particle Green function
G_G(t)_ij = G>(t)_ij = -i <ϕ|cᵢ(t), c†ⱼ(0)|ϕ>
where ϕ is a Slater determinant
"""
function G_G(t::Number, ϕ, ϵ::Vector{Float64}; kwargs...)
  N = length(ϵ)
  @assert size(ϕ) == (N, N)
  Nneg = count(en -> (en < 0), ϵ)
  Npart = get(kwargs, :Npart, Nneg)

  function compute_GG(i, j)
    gg = 0.0im
    for n in (Npart + 1):N
      gg += -im * ϕ[i, n] * conj(ϕ[j, n]) * exp(-im * ϵ[n] * t)
    end
    return gg
  end
  return _compute_correlator(N, compute_GG; kwargs...)
end

"""
Compute the lesser single-particle Green function
G_L(t)_ij = G<(t)_ij = +i <ϕ|c†ᵢ(0), cⱼ(t)|ϕ>
where ϕ is a Slater determinant
"""
function G_L(t::Number, ϕ, ϵ::Vector{Float64}; kwargs...)
  N = length(ϵ)
  @assert size(ϕ) == (N, N)
  Nneg = count(en -> (en < 0), ϵ)
  Npart = get(kwargs, :Npart, Nneg)

  function compute_GL(i, j)
    gl = 0.0im
    for n in 1:Npart
      gl += +im * ϕ[i, n] * conj(ϕ[j, n]) * exp(-im * ϵ[n] * t)
    end
    return gl
  end
  return _compute_correlator(N, compute_GL; kwargs...)
end
