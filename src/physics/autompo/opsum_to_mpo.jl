function svd_mpo(
  coefficient_type::Type,
  os::OpSum,
  sites::Vector{<:Index};
  mindim::Integer=1,
  maxdim::Integer=10000,
  cutoff::Real=1e-15,
  )
  N = length(sites)

  Vs = [Matrix{coefficient_type}(undef, 1, 1) for n in 1:N]
  temp_mpo = [MatElem{Scaled{coefficient_type,Prod{Op}}}[] for n in 1:N]

  rightmaps = fill(Dict{Vector{Op},Int}(), N)

  for n in 1:N
    leftbond_coefs = MatElem{coefficient_type}[]

    leftmap = Dict{Vector{Op},Int}()
    for term in os
      crosses_bond(term, n) || continue

      left = filter(t -> (only(site(t)) < n), terms(term))
      onsite = filter(t -> (only(site(t)) == n), terms(term))
      right = filter(t -> (only(site(t)) > n), terms(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = pos_in_link!(leftmap, left)
        bond_col = pos_in_link!(rightmaps[n - 1], vcat(onsite, right))
        bond_coef = convert(coefficient_type, coefficient(term))
        push!(leftbond_coefs, MatElem(bond_row, bond_col, bond_coef))
      end

      A_row = bond_col
      A_col = pos_in_link!(rightmaps[n], right)
      site_coef = one(coefficient_type)
      if A_row == -1
        site_coef = convert(coefficient_type, coefficient(term))
      end
      if isempty(onsite)
        if !using_auto_fermion() && isfermionic(right, sites)
          push!(onsite, Op("F", n))
        else
          push!(onsite, Op("Id", n))
        end
      end
      el = MatElem(A_row, A_col, site_coef * Prod(onsite))
      push!(temp_mpo[n], el)
    end
    remove_dups!(temp_mpo[n])
    if n > 1 && !isempty(leftbond_coefs)
      M = to_matrix(leftbond_coefs)
      U, S, V = svd(M)
      P = S .^ 2
      truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
      tdim = length(P)
      nc = size(M, 2)
      Vs[n - 1] = Matrix{coefficient_type}(V[1:nc, 1:tdim])
    end
  end

  llinks = Vector{Index{Int}}(undef, N + 1)
  llinks[1] = Index(2, "Link,l=0")

  H = MPO(sites)

  for n in 1:N
    VL = Matrix{coefficient_type}(undef, 1, 1)
    if n > 1
      VL = Vs[n - 1]
    end
    VR = Vs[n]
    tdim = isempty(rightmaps[n]) ? 0 : size(VR, 2)

    llinks[n + 1] = Index(2 + tdim, "Link,l=$n")

    ll = llinks[n]
    rl = llinks[n + 1]

    H[n] = ITensor()

    for el in temp_mpo[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coefficient(t)) > eps()) || continue

      M = zeros(coefficient_type, dim(ll), dim(rl))

      ct = convert(coefficient_type, coefficient(t))
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
      H[n] += T * compute_site_prod(sites, argument(t))
    end

    #
    # Special handling of starting and 
    # ending identity operators:
    #
    idM = zeros(coefficient_type, dim(ll), dim(rl))
    idM[1, 1] = 1.0
    idM[end, end] = 1.0
    T = itensor(idM, ll, rl)
    H[n] += T * compute_site_prod(sites, Prod([Op("Id", n)]))
  end

  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(llinks[N + 1])
  R[1] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end
