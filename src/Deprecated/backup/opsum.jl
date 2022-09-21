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

function svdMPO(ampo::OpSum, sites; kwargs...)::MPO
  mindim::Int = get(kwargs, :mindim, 1)
  maxdim::Int = get(kwargs, :maxdim, 10000)
  cutoff::Float64 = get(kwargs, :cutoff, 1E-15)

  N = length(sites)

  ValType = determineValType(data(ampo))

  Vs = [Matrix{ValType}(undef, 1, 1) for n in 1:N]
  tempMPO = [MatElem{MPOTerm}[] for n in 1:N]

  crosses_bond(t::MPOTerm, n::Int) = (site(ops(t)[1]) <= n <= site(ops(t)[end]))

  rightmap = Dict{OpTerm,Int}()
  next_rightmap = Dict{OpTerm,Int}()

  for n in 1:N
    leftbond_coefs = MatElem{ValType}[]

    leftmap = Dict{OpTerm,Int}()
    for term in data(ampo)
      crosses_bond(term, n) || continue

      left::OpTerm = filter(t -> (site(t) < n), ops(term))
      onsite::OpTerm = filter(t -> (site(t) == n), ops(term))
      right::OpTerm = filter(t -> (site(t) > n), ops(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap, left)
        bond_col = posInLink!(rightmap, mult(onsite, right))
        bond_coef = convert(ValType, coef(term))
        push!(leftbond_coefs, MatElem(bond_row, bond_col, bond_coef))
      end

      A_row = bond_col
      A_col = posInLink!(next_rightmap, right)
      site_coef = 1.0 + 0.0im
      if A_row == -1
        site_coef = coef(term)
      end
      if isempty(onsite)
        if !using_auto_fermion() && isfermionic(right, sites)
          push!(onsite, SiteOp("F", n))
        else
          push!(onsite, SiteOp("Id", n))
        end
      end
      el = MatElem(A_row, A_col, MPOTerm(site_coef, onsite))
      push!(tempMPO[n], el)
    end
    rightmap = next_rightmap
    next_rightmap = Dict{OpTerm,Int}()

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
    tdim = size(VR, 2)

    llinks[n + 1] = Index(2 + tdim, "Link,l=$n")

    ll = llinks[n]
    rl = llinks[n + 1]

    H[n] = ITensor()

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coef(t)) > eps()) || continue

      M = zeros(ValType, dim(ll), dim(rl))

      ct = convert(ValType, coef(t))
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
      H[n] += T * computeSiteProd(sites, ops(t))
    end

    #
    # Special handling of starting and 
    # ending identity operators:
    #
    idM = zeros(ValType, dim(ll), dim(rl))
    idM[1, 1] = 1.0
    idM[end, end] = 1.0
    T = itensor(idM, ll, rl)
    H[n] += T * computeSiteProd(sites, SiteOp[SiteOp("Id", n)])
  end

  L = ITensor(llinks[1])
  L[end] = 1.0

  R = ITensor(llinks[N + 1])
  R[1] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end #svdMPO

function qn_svdMPO(ampo::OpSum, sites; kwargs...)::MPO
  mindim::Int = get(kwargs, :mindim, 1)
  maxdim::Int = get(kwargs, :maxdim, 10000)
  cutoff::Float64 = get(kwargs, :cutoff, 1E-15)

  N = length(sites)

  ValType = determineValType(data(ampo))

  Vs = [Dict{QN,Matrix{ValType}}() for n in 1:(N + 1)]
  tempMPO = [QNMatElem{MPOTerm}[] for n in 1:N]

  crosses_bond(t::MPOTerm, n::Int) = (site(ops(t)[1]) <= n <= site(ops(t)[end]))

  rightmap = Dict{Pair{OpTerm,QN},Int}()
  next_rightmap = Dict{Pair{OpTerm,QN},Int}()

  # A cache of the ITensor operators on a certain site
  # of a certain type
  op_cache = Dict{Pair{String,Int},ITensor}()

  for n in 1:N
    leftbond_coefs = Dict{QN,Vector{MatElem{ValType}}}()

    leftmap = Dict{Pair{OpTerm,QN},Int}()
    for term in data(ampo)
      crosses_bond(term, n) || continue

      left::OpTerm = filter(t -> (site(t) < n), ops(term))
      onsite::OpTerm = filter(t -> (site(t) == n), ops(term))
      right::OpTerm = filter(t -> (site(t) > n), ops(term))

      function calcQN(term::OpTerm)
        q = QN()
        for st in term
          op_tensor = get(op_cache, name(st) => site(st), nothing)
          if op_tensor === nothing
            op_tensor = op(sites[site(st)], name(st); params(st)...)
            op_cache[name(st) => site(st)] = op_tensor
          end
          q -= flux(op_tensor)
        end
        return q
      end
      lqn = calcQN(left)
      sqn = calcQN(onsite)

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap, left => lqn)
        bond_col = posInLink!(rightmap, mult(onsite, right) => lqn)
        bond_coef = convert(ValType, coef(term))
        q_leftbond_coefs = get!(leftbond_coefs, lqn, MatElem{ValType}[])
        push!(q_leftbond_coefs, MatElem(bond_row, bond_col, bond_coef))
      end

      rqn = sqn + lqn
      A_row = bond_col
      A_col = posInLink!(next_rightmap, right => rqn)
      site_coef = 1.0 + 0.0im
      if A_row == -1
        site_coef = coef(term)
      end
      if isempty(onsite)
        if !using_auto_fermion() && isfermionic(right, sites)
          push!(onsite, SiteOp("F", n))
        else
          push!(onsite, SiteOp("Id", n))
        end
      end
      el = QNMatElem(lqn, rqn, A_row, A_col, MPOTerm(site_coef, onsite))
      push!(tempMPO[n], el)
    end
    rightmap = next_rightmap
    next_rightmap = Dict{Pair{OpTerm,QN},Int}()

    remove_dups!(tempMPO[n])

    if n > 1 && !isempty(leftbond_coefs)
      for (q, mat) in leftbond_coefs
        M = toMatrix(mat)
        U, S, V = svd(M)
        P = S .^ 2
        truncate!(P; maxdim=maxdim, cutoff=cutoff, mindim=mindim)
        tdim = length(P)
        nc = size(M, 2)
        Vs[n][q] = Matrix{ValType}(V[1:nc, 1:tdim])
      end
    end
  end

  #
  # Make MPO link indices
  #
  d0 = 2
  llinks = Vector{QNIndex}(undef, N + 1)
  # Set dir=In for fermionic ordering, avoid arrow sign
  # <fermions>:
  linkdir = using_auto_fermion() ? In : Out
  llinks[1] = Index(QN() => d0; tags="Link,l=0", dir=linkdir)
  for n in 1:N
    qi = Vector{Pair{QN,Int}}()
    if !haskey(Vs[n + 1], QN())
      # Make sure QN=zero is first in list of sectors
      push!(qi, QN() => d0)
    end
    for (q, Vq) in Vs[n + 1]
      cols = size(Vq, 2)
      if q == QN()
        # Make sure QN=zero is first in list of sectors
        insert!(qi, 1, q => d0 + cols)
      else
        if using_auto_fermion() # <fermions>
          push!(qi, (-q) => cols)
        else
          push!(qi, q => cols)
        end
      end
    end
    # Set dir=In for fermionic ordering, avoid arrow sign
    # <fermions>:
    llinks[n + 1] = Index(qi...; tags="Link,l=$n", dir=linkdir)
  end

  H = MPO(N)

  # Constants which define MPO start/end scheme
  startState = 2
  endState = 1

  for n in 1:N
    finalMPO = Dict{Tuple{QN,OpTerm},Matrix{ValType}}()

    ll = llinks[n]
    rl = llinks[n + 1]

    function defaultMat(ll, rl, lqn, rqn)
      #ldim = qnblockdim(ll,lqn)
      #rdim = qnblockdim(rl,rqn)
      ldim = blockdim(ll, lqn)
      rdim = blockdim(rl, rqn)
      return zeros(ValType, ldim, rdim)
    end

    idTerm = [SiteOp("Id", n)]
    finalMPO[(QN(), idTerm)] = defaultMat(ll, rl, QN(), QN())
    idM = finalMPO[(QN(), idTerm)]
    idM[1, 1] = 1.0
    idM[2, 2] = 1.0

    for el in tempMPO[n]
      t = el.val
      (abs(coef(t)) > eps()) || continue
      A_row = el.row
      A_col = el.col

      M = get!(finalMPO, (el.rowqn, ops(t)), defaultMat(ll, rl, el.rowqn, el.colqn))

      # rowShift and colShift account for
      # special entries in the zero-QN sector
      # of the MPO
      rowShift = (el.rowqn == QN()) ? 2 : 0
      colShift = (el.colqn == QN()) ? 2 : 0

      ct = convert(ValType, coef(t))
      if A_row == -1 && A_col == -1 #onsite term
        M[startState, endState] += ct
      elseif A_row == -1 #term starting on site n
        VR = Vs[n + 1][el.colqn]
        for c in 1:size(VR, 2)
          z = ct * VR[A_col, c]
          M[startState, colShift + c] += z
        end
      elseif A_col == -1 #term ending on site n
        VL = Vs[n][el.rowqn]
        for r in 1:size(VL, 2)
          z = ct * conj(VL[A_row, r])
          M[rowShift + r, endState] += z
        end
      else
        VL = Vs[n][el.rowqn]
        VR = Vs[n + 1][el.colqn]
        for r in 1:size(VL, 2), c in 1:size(VR, 2)
          z = ct * conj(VL[A_row, r]) * VR[A_col, c]
          M[rowShift + r, colShift + c] += z
        end
      end
    end

    s = sites[n]
    H[n] = ITensor()
    for (q_op, M) in finalMPO
      op_prod = q_op[2]
      Op = computeSiteProd(sites, op_prod)

      rq = q_op[1]
      sq = flux(Op)
      cq = rq - sq

      if using_auto_fermion()
        # <fermions>:
        # MPO is defined with Index order
        # of (rl,s[n]',s[n],cl) where rl = row link, cl = col link
        # so compute sign that would result by permuting cl from
        # second position to last position:
        if fparity(sq) == 1 && fparity(cq) == 1
          Op .*= -1
        end
      end

      rn = qnblocknum(ll, rq)
      cn = qnblocknum(rl, cq)

      #TODO: wrap following 3 lines into a function
      _block = Block(rn, cn)
      T = BlockSparseTensor(ValType, [_block], (dag(ll), rl))
      #blockview(T, _block) .= M
      T[_block] .= M

      IT = itensor(T)
      H[n] += IT * Op
    end
  end

  L = ITensor(llinks[1])
  L[startState] = 1.0

  R = ITensor(dag(llinks[N + 1]))
  R[endState] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end #qn_svdMPO

function sorteachterm!(ampo::OpSum, sites)
  ampo = copy(ampo)
  isless_site(o1::SiteOp, o2::SiteOp) = site(o1) < site(o2)
  N = length(sites)
  for t in data(ampo)
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
  return ampo
end

function check_numerical_opsum(ampo::OpSum)
  mpoterms = data(ampo)
  for mpoterm in mpoterms
    operators = ops(mpoterm)
    for operator in name.(operators)
      operator isa Array{<:Number} && return true
    end
  end
  return false
end

function sortmergeterms!(ampo::OpSum)
  check_numerical_opsum(ampo) && return ampo
  sort!(data(ampo))
  # Merge (add) terms with same operators
  da = data(ampo)
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

  setdata!(ampo, ndata)
  return ampo
end

"""
    MPO(ampo::OpSum,sites::Vector{<:Index};kwargs...)

Convert an OpSum object `ampo` to an
MPO, with indices given by `sites`. The
resulting MPO will have the indices
`sites[1], sites[1]', sites[2], sites[2]'`
etc. The conversion is done by an algorithm
that compresses the MPO resulting from adding
the OpSum terms together, often achieving
the minimum possible bond dimension.

# Examples

```julia
ampo = OpSum()
ampo += ("Sz",1,"Sz",2)
ampo += ("Sz",2,"Sz",3)
ampo += ("Sz",3,"Sz",4)

sites = siteinds("S=1/2",4)
H = MPO(ampo,sites)
```
"""
function MPO(ampo::OpSum, sites::Vector{<:Index}; kwargs...)::MPO
  length(data(ampo)) == 0 && error("OpSum has no terms")

  ampo = deepcopy(ampo)
  sorteachterm!(ampo, sites)
  sortmergeterms!(ampo)

  if hasqns(sites[1])
    return qn_svdMPO(ampo, sites; kwargs...)
  end
  return svdMPO(ampo, sites; kwargs...)
end
