# `ValType::Type{<:Number}` is used instead of `ValType::Type` for efficiency, possibly due to increased method specialization.
# See https://github.com/ITensor/ITensors.jl/pull/1183.
function svdMPO(ValType::Type{<:Number}, os::OpSum{C}, sites; kwargs...)::MPO where {C}
  mindim::Int = get(kwargs, :mindim, 1)
  maxdim::Int = get(kwargs, :maxdim, 10000)
  cutoff::Float64 = get(kwargs, :cutoff, 1E-15)

  N = length(sites)

  # Specifying the element type with `Matrix{ValType}[...]` improves type inference and therefore efficiency.
  # See https://github.com/ITensor/ITensors.jl/pull/1183.
  Vs = Matrix{ValType}[Matrix{ValType}(undef, 1, 1) for n in 1:N]
  tempMPO = [MatElem{Scaled{C,Prod{Op}}}[] for n in 1:N]

  function crosses_bond(t::Scaled{C,Prod{Op}}, n::Int) where {C}
    return (only(site(t[1])) <= n <= only(site(t[end])))
  end

  rightmaps = fill(Dict{Vector{Op},Int}(), N)

  for n in 1:N
    leftbond_coefs = MatElem{ValType}[]

    leftmap = Dict{Vector{Op},Int}()
    for term in os
      crosses_bond(term, n) || continue

      left = filter(t -> (only(site(t)) < n), terms(term))
      onsite = filter(t -> (only(site(t)) == n), terms(term))
      right = filter(t -> (only(site(t)) > n), terms(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap, left)
        bond_col = posInLink!(rightmaps[n - 1], vcat(onsite, right))
        bond_coef = convert(ValType, coefficient(term))
        push!(leftbond_coefs, MatElem(bond_row, bond_col, bond_coef))
      end

      A_row = bond_col
      A_col = posInLink!(rightmaps[n], right)
      site_coef = one(C)
      if A_row == -1
        site_coef = coefficient(term)
      end
      if isempty(onsite)
        if !using_auto_fermion() && isfermionic(right, sites)
          push!(onsite, Op("F", n))
        else
          push!(onsite, Op("Id", n))
        end
      end
      el = MatElem(A_row, A_col, site_coef * Prod(onsite))
      push!(tempMPO[n], el)
    end
    remove_dups!(tempMPO[n])
    if n > 1 && !isempty(leftbond_coefs)
      M = toMatrix(leftbond_coefs)
      U, S, V = svd(M)
      P = S .^ 2
      truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
      tdim = length(P)
      nc = size(M, 2)
      Vs[n - 1] = Matrix{ValType}(V[1:nc, 1:tdim])
    end
  end

  llinks = Vector{Index{Int}}(undef, N + 1)
  llinks[1] = Index(2, "Link,l=0")

  H = MPO(sites)

  for n in 1:N
    VL = Matrix{ValType}(undef, 1, 1)
    if n > 1
      VL = Vs[n - 1]
    end
    VR = Vs[n]
    tdim = isempty(rightmaps[n]) ? 0 : size(VR, 2)

    llinks[n + 1] = Index(2 + tdim, "Link,l=$n")

    ll = llinks[n]
    rl = llinks[n + 1]

    H[n] = ITensor()

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coefficient(t)) > eps()) || continue

      M = zeros(ValType, dim(ll), dim(rl))

      ct = convert(ValType, coefficient(t))
      if A_row == -1 && A_col == -1 #onsite term
        M[end, 1] += ct
      elseif A_row == -1 #term starting on site n
        for c in 1:size(VR, 2)
          z = ct * VR[A_col, c]
          M[end, 1 + c] += z
        end
      elseif A_col == -1 #term ending on site n
        for r in 1:size(VL, 2)
          z = ct * conj(VL[A_row, r])
          M[1 + r, 1] += z
        end
      else
        for r in 1:size(VL, 2), c in 1:size(VR, 2)
          z = ct * conj(VL[A_row, r]) * VR[A_col, c]
          M[1 + r, 1 + c] += z
        end
      end

      T = itensor(M, ll, rl)
      H[n] += T * computeSiteProd(sites, argument(t))
    end

    #
    # Special handling of starting and
    # ending identity operators:
    #
    idM = zeros(ValType, dim(ll), dim(rl))
    idM[1, 1] = 1.0
    idM[end, end] = 1.0
    T = itensor(idM, ll, rl)
    H[n] += T * computeSiteProd(sites, Prod([Op("Id", n)]))
  end

  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(llinks[N + 1])
  R[1] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end #svdMPO

function svdMPO(os::OpSum{C}, sites; kwargs...)::MPO where {C}
  # Function barrier to improve type stability
  ValType = determineValType(terms(os))
  return svdMPO(ValType, os, sites; kwargs...)
end
