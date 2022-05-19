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
