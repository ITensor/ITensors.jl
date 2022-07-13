function qn_svdMPO(os::OpSum, sites; kwargs...)::MPO
  mindim::Int = get(kwargs, :mindim, 1)
  maxdim::Int = get(kwargs, :maxdim, typemax(Int))
  cutoff::Float64 = get(kwargs, :cutoff, 1E-15)

  N = length(sites)

  ValType = determineValType(data(os))

  Vs = [Dict{QN,Matrix{ValType}}() for n in 1:(N + 1)]
  sparse_MPO = [QNMatElem{MPOTerm}[] for n in 1:N]

  crosses_bond(t::MPOTerm, n::Int) = (site(ops(t)[1]) <= n <= site(ops(t)[end]))

  # A cache of the ITensor operators on a certain site
  # of a certain type
  op_cache = Dict{Pair{String,Int},ITensor}()
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

  Hflux = -calcQN(ops(first(data(os))))

  rightmap = Dict{Pair{OpTerm,QN},Int}()
  next_rightmap = Dict{Pair{OpTerm,QN},Int}()

  for n in 1:N
    h_sparse = Dict{QN,Vector{MatElem{ValType}}}()

    leftmap = Dict{Pair{OpTerm,QN},Int}()
    for term in data(os)
      crosses_bond(term, n) || continue

      left::OpTerm = filter(t -> (site(t) < n), ops(term))
      onsite::OpTerm = filter(t -> (site(t) == n), ops(term))
      right::OpTerm = filter(t -> (site(t) > n), ops(term))

      lqn = calcQN(left)
      sqn = calcQN(onsite)

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap, left => lqn)
        bond_col = posInLink!(rightmap, mult(onsite, right) => lqn)
        bond_coef = convert(ValType, coef(term))
        q_h_sparse = get!(h_sparse, lqn, MatElem{ValType}[])
        push!(q_h_sparse, MatElem(bond_row, bond_col, bond_coef))
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
      push!(sparse_MPO[n], el)
    end
    remove_dups!(sparse_MPO[n])

    if n > 1 && !isempty(h_sparse)
      for (q, mat) in h_sparse
        h = toMatrix(mat)
        U, S, V = svd(h)
        P = S .^ 2
        truncate!(P; maxdim, cutoff, mindim)
        tdim = length(P)
        Vs[n][q] = Matrix{ValType}(V[:, 1:tdim])
      end
    end

    rightmap = next_rightmap
    next_rightmap = Dict{Pair{OpTerm,QN},Int}()
  end

  #
  # Make MPO link indices
  #
  llinks = Vector{QNIndex}(undef, N + 1)
  # Set dir=In for fermionic ordering, avoid arrow sign
  # <fermions>:
  linkdir = using_auto_fermion() ? In : Out
  llinks[1] = Index([QN() => 1, Hflux => 1]; tags="Link,l=0", dir=linkdir)
  for n in 1:N
    qi = Vector{Pair{QN,Int}}()
    push!(qi, QN() => 1)
    for (q, Vq) in Vs[n + 1]
      cols = size(Vq, 2)
      if using_auto_fermion() # <fermions>
        push!(qi, (-q) => cols)
      else
        push!(qi, q => cols)
      end
    end
    push!(qi, Hflux => 1)
    llinks[n + 1] = Index(qi...; tags="Link,l=$n", dir=linkdir)
  end

  H = MPO(N)

  # Find location where block of Index i
  # matches QN q, but *not* 1 or dim(i)
  # which are special ending/starting states
  function qnblock(i::Index, q::QN)
    for b in 2:(nblocks(i) - 1)
      flux(i, Block(b)) == q && return b
    end
    return error("Could not find block of QNIndex with matching QN")
  end
  qnblockdim(i::Index, q::QN) = blockdim(i, qnblock(i, q))

  for n in 1:N
    ll = llinks[n]
    rl = llinks[n + 1]

    begin_block = Dict{Tuple{QN,OpTerm},Matrix{ValType}}()
    cont_block = Dict{Tuple{QN,OpTerm},Matrix{ValType}}()
    end_block = Dict{Tuple{QN,OpTerm},Matrix{ValType}}()
    onsite_block = Dict{Tuple{QN,OpTerm},Matrix{ValType}}()

    for el in sparse_MPO[n]
      t = el.val
      (abs(coef(t)) > eps()) || continue
      A_row = el.row
      A_col = el.col
      ct = convert(ValType, coef(t))

      ldim = (A_row == -1) ? 1 : qnblockdim(ll, el.rowqn)
      rdim = (A_col == -1) ? 1 : qnblockdim(rl, el.colqn)
      zero_mat() = zeros(ValType, ldim, rdim)

      if A_row == -1 && A_col == -1
        # Onsite term
        M = get!(onsite_block, (el.rowqn, ops(t)), zeros(ValType, 1, 1))
        M[1, 1] += ct
      elseif A_row == -1
        # Operator beginning a term on site n
        M = get!(begin_block, (el.rowqn, ops(t)), zero_mat())
        VR = Vs[n + 1][el.colqn]
        for c in 1:size(VR, 2)
          M[1, c] += ct * VR[A_col, c]
        end
      elseif A_col == -1
        # Operator ending a term on site n
        M = get!(end_block, (el.rowqn, ops(t)), zero_mat())
        VL = Vs[n][el.rowqn]
        for r in 1:size(VL, 2)
          M[r, 1] += ct * conj(VL[A_row, r])
        end
      else
        # Operator continuing a term on site n
        M = get!(cont_block, (el.rowqn, ops(t)), zero_mat())
        VL = Vs[n][el.rowqn]
        VR = Vs[n + 1][el.colqn]
        for r in 1:size(VL, 2), c in 1:size(VR, 2)
          M[r, c] += ct * conj(VL[A_row, r]) * VR[A_col, c]
        end
      end
    end

    H[n] = ITensor()

    # Helper functions to compute block locations
    # of various blocks within the onsite blocks,
    # begin blocks, etc.
    loc_onsite(rq, cq) = Block(nblocks(ll), 1)
    loc_begin(rq, cq) = Block(nblocks(ll), qnblock(rl, cq))
    loc_cont(rq, cq) = Block(qnblock(ll, rq), qnblock(rl, cq))
    loc_end(rq, cq) = Block(qnblock(ll, rq), 1)

    for (loc, block) in (
      (loc_onsite, onsite_block),
      (loc_begin, begin_block),
      (loc_end, end_block),
      (loc_cont, cont_block),
    )
      for (q_op, M) in block
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

        b = loc(rq, cq)
        T = BlockSparseTensor(ValType, [b], (dag(ll), rl))
        T[b] .= M

        H[n] += (itensor(T) * Op)
      end
    end

    # Put in ending identity operator
    Id = op("Id", sites[n])
    b = Block(1, 1)
    T = BlockSparseTensor(ValType, [b], (dag(ll), rl))
    T[b] = 1
    H[n] += (itensor(T) * Id)

    # Put in starting identity operator
    b = Block(nblocks(ll), nblocks(rl))
    T = BlockSparseTensor(ValType, [b], (dag(ll), rl))
    T[b] = 1
    H[n] += (itensor(T) * Id)
  end # for n in 1:N

  L = ITensor(llinks[1])
  L[llinks[1] => end] = 1.0
  H[1] *= L

  R = ITensor(dag(llinks[N + 1]))
  R[dag(llinks[N + 1]) => 1] = 1.0
  H[N] *= R

  return H
end #qn_svdMPO
