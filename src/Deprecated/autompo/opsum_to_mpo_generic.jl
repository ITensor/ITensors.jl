isempty(op_qn::Pair{OpTerm,QN}) = isempty(op_qn.first)

# the key type is OpTerm for the dense case
# and is Pair{OpTerm,QN} for the QN conserving case
function posInLink!(linkmap::Dict{K,Int}, k::K)::Int where {K}
  isempty(k) && return -1
  pos = get(linkmap, k, -1)
  if pos == -1
    pos = length(linkmap) + 1
    linkmap[k] = pos
  end
  return pos
end

function determineValType(terms::Vector{MPOTerm})
  for t in terms
    (!isreal(coef(t))) && return ComplexF64
  end
  return Float64
end

function computeSiteProd(sites, ops::OpTerm)::ITensor
  i = site(ops[1])
  T = op(sites[i], ops[1].name; ops[1].params...)
  for j in 2:length(ops)
    (site(ops[j]) != i) && error("Mismatch of site number in computeSiteProd")
    opj = op(sites[i], ops[j].name; ops[j].params...)
    T = product(T, opj)
  end
  return T
end

function remove_dups!(v::Vector{T}) where {T}
  N = length(v)
  (N == 0) && return nothing
  sort!(v)
  n = 1
  u = 2
  while u <= N
    while u < N && v[u] == v[n]
      u += 1
    end
    if v[u] != v[n]
      v[n + 1] = v[u]
      n += 1
    end
    u += 1
  end
  resize!(v, n)
  return nothing
end #remove_dups!

function sorteachterm!(os::OpSum, sites)
  os = copy(os)
  isless_site(o1::SiteOp, o2::SiteOp) = site(o1) < site(o2)
  N = length(sites)
  for t in data(os)
    Nt = length(t.ops)
    prevsite = N + 1 #keep track of whether we are switching
    #to a new site to make sure F string
    #is only placed at most once for each site

    # Sort operators in t by site order,
    # and keep the permutation used, perm, for analysis below
    perm = Vector{Int}(undef, Nt)
    sortperm!(perm, t.ops; alg=InsertionSort, lt=isless_site)

    t.ops = t.ops[perm]

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    # and inserting string "F" operators
    parity = +1
    for n in Nt:-1:1
      currsite = site(t.ops[n])
      fermionic = has_fermion_string(name(t.ops[n]), sites[site(t.ops[n])])
      if !using_auto_fermion() && (parity == -1) && (currsite < prevsite)
        # Put local piece of Jordan-Wigner string emanating
        # from fermionic operators to the right
        # (Remaining F operators will be put in by svdMPO)
        t.ops[n] = SiteOp("$(name(t.ops[n])) * F", site(t.ops[n]))
      end
      prevsite = currsite

      if fermionic
        parity = -parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[n] = 0
      end
    end
    if parity == -1
      error("Parity-odd fermionic terms not yet supported by AutoMPO")
    end

    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero, perm)
    # and account for anti-commuting, fermionic operators 
    # during above sort; put resulting sign into coef
    t.coef *= parity_sign(perm)
  end
  return os
end

function check_numerical_opsum(os::OpSum)
  mpoterms = data(os)
  for mpoterm in mpoterms
    operators = ops(mpoterm)
    for operator in name.(operators)
      operator isa Array{<:Number} && return true
    end
  end
  return false
end

function sortmergeterms!(os::OpSum)
  check_numerical_opsum(os) && return os
  sort!(data(os))
  # Merge (add) terms with same operators
  da = data(os)
  ndata = MPOTerm[]
  last_term = copy(da[1])
  for n in 2:length(da)
    if ops(da[n]) == ops(last_term)
      last_term.coef += coef(da[n])
    else
      push!(ndata, last_term)
      last_term = copy(da[n])
    end
  end
  push!(ndata, last_term)

  setdata!(os, ndata)
  return os
end

"""
    MPO(os::OpSum,sites::Vector{<:Index};kwargs...)
       
Convert an OpSum object `os` to an
MPO, with indices given by `sites`. The
resulting MPO will have the indices
`sites[1], sites[1]', sites[2], sites[2]'`
etc. The conversion is done by an algorithm
that compresses the MPO resulting from adding
the OpSum terms together, often achieving
the minimum possible bond dimension.

# Examples
```julia
os = OpSum()
os += ("Sz",1,"Sz",2)
os += ("Sz",2,"Sz",3)
os += ("Sz",3,"Sz",4)

sites = siteinds("S=1/2",4)
H = MPO(os,sites)
```
"""
function MPO(os::OpSum, sites::Vector{<:Index}; kwargs...)::MPO
  length(data(os)) == 0 && error("OpSum has no terms")

  os = deepcopy(os)
  sorteachterm!(os, sites)
  sortmergeterms!(os)

  if hasqns(sites[1])
    return qn_svdMPO(os, sites; kwargs...)
  end
  return svdMPO(os, sites; kwargs...)
end
