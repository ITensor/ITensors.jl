#
# Optimizations:
#  - replace leftmap, rightmap with sorted vectors
# 

###########################
# SiteOp                  # 
###########################

struct SiteOp
  name::String
  site::Int
end

convert(::Type{SiteOp}, op::Pair{String, Int}) =
  SiteOp(first(op), last(op))

name(s::SiteOp) = s.name
site(s::SiteOp) = s.site

show(io::IO,s::SiteOp) = print(io,"\"$(name(s))\"($(site(s)))")

(s1::SiteOp == s2::SiteOp) =
  (s1.site == s2.site && s1.name == s2.name)

function isless(s1::SiteOp, s2::SiteOp)
  if site(s1) != site(s2)
    return site(s1) < site(s2)
  end
  return name(s1) < name(s2)
end

###########################
# OpTerm                  # 
###########################

const OpTerm = Vector{SiteOp}

function (o1::OpTerm == o2::OpTerm)
  (length(o1)==length(o2)) || return false
  @inbounds for n=1:length(o1)
    (o1[n]!=o2[n]) && return false
  end
  return true
end

function isless(o1::OpTerm, o2::OpTerm)
  if length(o1) != length(o2) 
    return length(o1) < length(o2)
  end
  for n=1:length(o1)
    if o1[n]!=o2[n]
      return (o1[n] < o2[n])
    end
  end
  return false
end

mult(t1::OpTerm,t2::OpTerm) = isempty(t2) ? t1 : vcat(t1,t2)

function isfermionic(t::OpTerm, sites)::Bool
  p = +1
  for op in t
    if has_fermion_string(name(op), sites[site(op)])
      p *= -1
    end
  end
  return (p == -1)
end

###########################
# MPOTerm                 # 
###########################

mutable struct MPOTerm
  coef::ComplexF64
  ops::OpTerm
end
coef(op::MPOTerm) = op.coef
ops(op::MPOTerm) = op.ops

copy(t::MPOTerm) = MPOTerm(coef(t),copy(ops(t)))

function (t1::MPOTerm == t2::MPOTerm)
  return coef(t1) ≈ coef(t2) && ops(t1) == ops(t2)
end

function isless(t1::MPOTerm, t2::MPOTerm)
  if ops(t1) == ops(t2)
    if coef(t1) ≈ coef(t2)
      return false
    else
      ct1 = coef(t1)
      ct2 = coef(t2)
      #"lexicographic" ordering on  complex numbers
      return real(ct1) < real(ct2) || 
             (real(ct1) ≈ real(ct2) && imag(ct1) < imag(ct2))
    end
  end
  return ops(t1) < ops(t2)
end

function MPOTerm(c::Number,
                 op1::String,
                 i1::Int) 
  return MPOTerm(convert(ComplexF64,c),[SiteOp(op1,i1)])
end

function MPOTerm(c::Number,
                 op1::String,i1::Int,
                 op2::String,i2::Int)
  return MPOTerm(convert(ComplexF64,c),[SiteOp(op1,i1),SiteOp(op2,i2)])
end

function MPOTerm(c::Number,
                 op1::String,i1::Int,
                 op2::String,i2::Int,
                 ops...)
  vop = OpTerm(undef,2+div(length(ops),2))
  vop[1] = SiteOp(op1,i1)
  vop[2] = SiteOp(op2,i2)
  for n = 1:div(length(ops),2)
    vop[2+n] = SiteOp(ops[2*n-1],ops[2*n])
  end
  return MPOTerm(convert(ComplexF64,c),vop)
end

#function MPOTerm(c::Number,
#                 ops::OpTerm)
#  return MPOTerm(convert(ComplexF64,c),ops)
#end

function Base.show(io::IO,
                   op::MPOTerm) 
  c = coef(op)
  if iszero(imag(c))
    print(io,"$(real(c)) ")
  elseif iszero(real(c))
    print(io,"$(imag(c))im ")
  else
    print(io,"($c) ")
  end
  for o in ops(op)
    print(io,"\"$(name(o))\"($(site(o))) ")
  end
end

############################
## AutoMPO                 #
############################

"""
An AutoMPO stores a collection of
operator terms, to be later summed
together into an MPO by calling
the function `MPO` on the AutoMPO object. 
Each term is a product of local operators
specified by names such as "Sz" or "N",
times an optional coefficient which
can be real or complex.

Which local operator names are available
is determined by the function `op`
associated with the TagType defined by
special Index tags, such as "S=1/2","S=1",
"Fermion", and "Electron".
"""
mutable struct AutoMPO
  data::Vector{MPOTerm}
  AutoMPO(terms::Vector{MPOTerm}) = new(terms)
end

"""
    AutoMPO()
    
Construct an empty AutoMPO
"""
AutoMPO() = AutoMPO(Vector{MPOTerm}())

data(ampo::AutoMPO) = ampo.data
setdata!(ampo::AutoMPO,ndata) = (ampo.data = ndata)

push!(ampo::AutoMPO, term) = push!(data(ampo), term)

Base.:(==)(ampo1::AutoMPO,
           ampo2::AutoMPO) = data(ampo1) == data(ampo2)

Base.copy(ampo::AutoMPO) = AutoMPO(copy(data(ampo)))

Base.size(ampo::AutoMPO) = size(data(ampo))


"""
    add!(ampo::AutoMPO,
         op1::String, i1::Int)

    add!(ampo::AutoMPO,
         coef::Number,
         op1::String, i1::Int)

    add!(ampo::AutoMPO,
         op1::String, i1::Int,
         op2::String, i2::Int,
         ops...)

    add!(ampo::AutoMPO,
         coef::Number,
         op1::String, i1::Int,
         op2::String, i2::Int,
         ops...)

    +(ampo:AutoMPO, term::Tuple)

Add a single- or multi-site operator 
term to the AutoMPO `ampo`. Each operator
is specified by a name (String) and a
site number (Int). The second version
accepts a real or complex coefficient.

The `+` operator version of this function
accepts a tuple with entries either
(String,Int,String,Int,...) or
(Number,String,Int,String,Int,...)
where these tuple values are the same
as valid inputs to the `add!` function.
For inputting a very large number of
terms (tuples) to an AutoMPO, consider
using the broadcasted operator `.+=`
which avoids reallocating the AutoMPO
after each addition.

# Examples
```julia
ampo = AutoMPO()

add!(ampo,"Sz",2,"Sz",3)

ampo += ("Sz",3,"Sz",4)

ampo += (0.5,"S+",4,"S-",5)

ampo .+= (0.5,"S+",5,"S-",6)
```
"""
function add!(ampo::AutoMPO,
              coef::Number,
              op::String, i::Int)
  push!(data(ampo),MPOTerm(coef,op,i))
  return
end

add!(ampo::AutoMPO,op::String, i::Int) = add!(ampo,1.0,op,i)


function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int)
  push!(data(ampo),MPOTerm(coef,op1,i1,op2,i2))
  return
end

add!(ampo::AutoMPO, op1::String, i1::Int, op2::String, i2::Int) =
  add!(ampo,1.0,op1,i1,op2,i2)

function add!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)
  push!(ampo, MPOTerm(coef, op1, i1, op2, i2, ops...))
  return ampo
end

function add!(ampo::AutoMPO, op1::String, i1::Int,
              op2::String, i2::Int, ops...)
  return add!(ampo, 1.0, op1, i1, op2, i2, ops...)
end

function add!(ampo::AutoMPO, ops::Vector{Pair{String,Int64}})
  push!(ampo, MPOTerm(1.0, ops))
  return ampo
end

"""
    subtract!(ampo::AutoMPO,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)

    subtract!(ampo::AutoMPO,
              coef::Number,
              op1::String, i1::Int,
              op2::String, i2::Int,
              ops...)

Subtract a multi-site operator term
from the AutoMPO `ampo`. Each operator
is specified by a name (String) and a
site number (Int). The second version
accepts a real or complex coefficient.
"""
subtract!(ampo::AutoMPO,
          op1::String, i1::Int,
          op2::String, i2::Int,
          ops...) = add!(ampo, -1.0, op1, i1, op2, i2, ops...)

function subtract!(ampo::AutoMPO,
                   coef::Number,
                   op1::String, i1::Int,
                   op2::String, i2::Int,
                   ops...)
  push!(ampo, -MPOTerm(coef, op1, i1, op2, i2, ops...))
  return ampo
end

-(t::MPOTerm) = MPOTerm(-coef(t), ops(t))

function (ampo::AutoMPO + term::Tuple)
  ampo_plus_term = copy(ampo)
  add!(ampo_plus_term, term...)
  return ampo_plus_term
end

function (ampo::AutoMPO + term::Vector{Pair{String,Int64}})
  ampo_plus_term = copy(ampo)
  add!(ampo_plus_term, term)
  return ampo_plus_term
end

function (ampo::AutoMPO - term::Tuple)
  ampo_plus_term = copy(ampo)
  subtract!(ampo_plus_term, term...)
  return ampo_plus_term
end

#
# ampo .+= ("Sz",1) syntax using broadcasting
#

struct AutoMPOStyle <: Broadcast.BroadcastStyle end
Base.BroadcastStyle(::Type{<:AutoMPO}) = AutoMPOStyle()

struct AutoMPOAddTermStyle <: Broadcast.BroadcastStyle end

Base.broadcastable(ampo::AutoMPO) = ampo

Base.BroadcastStyle(::AutoMPOStyle,
                    ::Broadcast.Style{Tuple}) = AutoMPOAddTermStyle()

Broadcast.instantiate(bc::Broadcast.Broadcasted{AutoMPOAddTermStyle}) = bc

function Base.copyto!(ampo,
                      bc::Broadcast.Broadcasted{AutoMPOAddTermStyle,
                                                <:Any,
                                                typeof(+)})
  add!(ampo, bc.args[2]...)
  return ampo
end

#
# ampo .-= ("Sz",1) syntax using broadcasting
#

function Base.copyto!(ampo,
                      bc::Broadcast.Broadcasted{AutoMPOAddTermStyle,
                                                <:Any,
                                                typeof(-)})
  subtract!(ampo, bc.args[2]...)
  return ampo
end

function Base.show(io::IO,
                   ampo::AutoMPO) 
  println(io,"AutoMPO:")
  for term in data(ampo)
    println(io,"  $term")
  end
end

##################################
# MatElem (simple sparse matrix) #
##################################

struct MatElem{T}
  row::Int
  col::Int
  val::T
end

#function Base.show(io::IO,m::MatElem)
#  print(io,"($(m.row),$(m.col),$(m.val))")
#end

function toMatrix(els::Vector{MatElem{T}})::Matrix{T} where {T}
  nr = 0
  nc = 0
  for el in els
    nr = max(nr,el.row)
    nc = max(nc,el.col)
  end
  M = zeros(T,nr,nc)
  for el in els
    M[el.row,el.col] = el.val
  end
  return M
end

function Base.:(==)(m1::MatElem{T},m2::MatElem{T})::Bool where {T}
  return (m1.row==m2.row && m1.col==m2.col && m1.val==m2.val)
end

function Base.isless(m1::MatElem{T},m2::MatElem{T})::Bool where {T}
  if m1.row != m2.row
    return m1.row < m2.row
  elseif m1.col != m2.col
    return m1.col < m2.col
  end
  return m1.val < m2.val
end

struct QNMatElem{T}
  rowqn::QN
  colqn::QN
  row::Int
  col::Int
  val::T
end

function Base.:(==)(m1::QNMatElem{T},m2::QNMatElem{T})::Bool where {T}
  return (m1.row==m2.row && m1.col==m2.col && m1.val==m2.val && m1.rowqn==m2.rowqn && m1.colqn==m2.colqn)
end

function Base.isless(m1::QNMatElem{T},m2::QNMatElem{T})::Bool where {T}
  if m1.rowqn != m2.rowqn
    return m1.rowqn < m2.rowqn
  elseif m1.colqn != m2.colqn
    return m1.colqn < m2.colqn
  elseif m1.row != m2.row
    return m1.row < m2.row
  elseif m1.col != m2.col
    return m1.col < m2.col
  end
  return m1.val < m2.val
end

isempty(op_qn::Pair{OpTerm,QN}) = isempty(op_qn.first)

# the key type is OpTerm for the dense case
# and is Pair{OpTerm,QN} for the QN conserving case
function posInLink!(linkmap::Dict{K,Int},
                    k::K)::Int where {K}
  isempty(k) && return -1
  pos = get(linkmap,k,-1)
  if pos == -1
    pos = length(linkmap)+1
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

function computeSiteProd(sites,
                         ops::OpTerm)::ITensor
  i = ops[1].site
  T = op(sites[i],ops[1].name)
  for j=2:length(ops)
    (ops[j].site != i) && error("Mismatch of site number in computeSiteProd")
    opj = op(sites[i],ops[j].name)
    T = product(T, opj)
  end
  return T
end


function remove_dups!(v::Vector{T}) where {T}
  N = length(v)
  (N==0) && return
  sort!(v)
  n = 1
  u = 2
  while u <= N
    while u < N && v[u]==v[n] 
      u += 1
    end
    if v[u] != v[n]
      v[n+1] = v[u]
      n += 1
    end
    u += 1
  end
  resize!(v,n)
  return
end #remove_dups!


function svdMPO(ampo::AutoMPO,
                sites; 
                kwargs...)::MPO

  mindim::Int = get(kwargs,:mindim,1)
  maxdim::Int = get(kwargs,:maxdim,10000)
  cutoff::Float64 = get(kwargs,:cutoff,1E-13)

  N = length(sites)

  ValType = determineValType(data(ampo))

  Vs = [Matrix{ValType}(undef,1,1) for n=1:N]
  tempMPO = [MatElem{MPOTerm}[] for n=1:N]

  crosses_bond(t::MPOTerm,n::Int) = (ops(t)[1].site <= n <= ops(t)[end].site)

  rightmap = Dict{OpTerm,Int}()
  next_rightmap = Dict{OpTerm,Int}()
  
  for n=1:N

    leftbond_coefs = MatElem{ValType}[]

    leftmap = Dict{OpTerm,Int}()
    for term in data(ampo)
      crosses_bond(term,n) || continue

      left::OpTerm   = filter(t->(t.site < n),ops(term))
      onsite::OpTerm = filter(t->(t.site == n),ops(term))
      right::OpTerm  = filter(t->(t.site > n),ops(term))

      bond_row = -1
      bond_col = -1
      if !isempty(left)
        bond_row = posInLink!(leftmap,left)
        bond_col = posInLink!(rightmap,mult(onsite,right))
        bond_coef = convert(ValType,coef(term))
        push!(leftbond_coefs,MatElem(bond_row,bond_col,bond_coef))
      end

      A_row = bond_col
      A_col = posInLink!(next_rightmap,right)
      site_coef = 1.0+0.0im
      if A_row == -1
        site_coef = coef(term)
      end
      if isempty(onsite)
        if isfermionic(right, sites)
          push!(onsite,SiteOp("F",n))
        else
          push!(onsite,SiteOp("Id",n))
        end
      end
      el = MatElem(A_row,A_col,MPOTerm(site_coef,onsite))
      push!(tempMPO[n],el)
    end
    rightmap = next_rightmap
    next_rightmap = Dict{OpTerm,Int}()

    remove_dups!(tempMPO[n])

    if n > 1 && !isempty(leftbond_coefs)
      M = toMatrix(leftbond_coefs)
      U,S,V = svd(M)
      P = S.^2
      truncate!(P;maxdim=maxdim,cutoff=cutoff,mindim=mindim)
      tdim = length(P)
      nc = size(M,2)
      Vs[n-1] = Matrix{ValType}(V[1:nc,1:tdim])
    end

  end

  llinks = Vector{Index{Int}}(undef, N+1)
  llinks[1] = Index(2, "Link,l=0")

  H = MPO(sites)

  # Constants which define MPO start/end scheme
  rowShift = 2
  colShift = 2
  startState = 2
  endState = 1

  for n=1:N
    VL = Matrix{ValType}(undef,1,1)
    if n > 1
      VL = Vs[n-1]
    end
    VR = Vs[n]
    tdim = size(VR,2)

    llinks[n+1] = Index(2+tdim,"Link,l=$n")

    finalMPO = Dict{OpTerm,Matrix{ValType}}()

    ll = llinks[n]
    rl = llinks[n+1]

    idTerm = [SiteOp("Id",n)]
    finalMPO[idTerm] = zeros(ValType,dim(ll),dim(rl))
    idM = finalMPO[idTerm]
    idM[1,1] = 1.0
    idM[2,2] = 1.0

    defaultMat() = zeros(ValType,dim(ll),dim(rl))

    for el in tempMPO[n]
      A_row = el.row
      A_col = el.col
      t = el.val
      (abs(coef(t)) > eps()) || continue

      M = get!(finalMPO,ops(t),defaultMat())

      ct = convert(ValType,coef(t))
      if A_row==-1 && A_col==-1 #onsite term
        M[startState,endState] += ct
      elseif A_row==-1 #term starting on site n
        for c=1:size(VR,2)
          z = ct*VR[A_col,c]
          M[startState,colShift+c] += z
        end
      elseif A_col==-1 #term ending on site n
        for r=1:size(VL,2)
          z = ct*conj(VL[A_row,r])
          M[rowShift+r,endState] += z
        end
      else
        for r=1:size(VL,2),c=1:size(VR,2)
          z = ct*conj(VL[A_row,r])*VR[A_col,c]
          M[rowShift+r,colShift+c] += z
        end
      end
    end

    s = sites[n]
    #H[n] = emptyITensor(dag(s),s',ll,rl)
    H[n] = emptyITensor()
    for (op,M) in finalMPO
      T = itensor(M,ll,rl)
      H[n] += T*computeSiteProd(sites,op)
    end

  end

  L = emptyITensor(llinks[1])
  L[startState] = 1.0

  R = emptyITensor(llinks[N+1])
  R[endState] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end #svdMPO

function qn_svdMPO(ampo::AutoMPO,
                   sites; 
                   kwargs...)::MPO

  mindim::Int = get(kwargs,:mindim,1)
  maxdim::Int = get(kwargs,:maxdim,10000)
  cutoff::Float64 = get(kwargs,:cutoff,1E-13)

  N = length(sites)

  ValType = determineValType(data(ampo))

  Vs = [Dict{QN,Matrix{ValType}}() for n=1:N+1]
  tempMPO = [QNMatElem{MPOTerm}[] for n=1:N]

  crosses_bond(t::MPOTerm,n::Int) = (ops(t)[1].site <= n <= ops(t)[end].site)

  rightmap = Dict{Pair{OpTerm,QN},Int}()
  next_rightmap = Dict{Pair{OpTerm,QN},Int}()
  
  # A cache of the ITensor operators on a certain site
  # of a certain type
  op_cache = Dict{Pair{String, Int}, ITensor}()

  for n=1:N

    leftbond_coefs = Dict{QN,Vector{MatElem{ValType}}}()

    leftmap = Dict{Pair{OpTerm,QN},Int}()
    for term in data(ampo)
      crosses_bond(term,n) || continue

      left::OpTerm   = filter(t->(t.site < n),ops(term))
      onsite::OpTerm = filter(t->(t.site == n),ops(term))
      right::OpTerm  = filter(t->(t.site > n),ops(term))

      function calcQN(term::OpTerm)
        q = QN()
        for st in term
          op_tensor = get(op_cache, name(st) => site(st), nothing)
          if op_tensor === nothing
            op_tensor = op(sites[site(st)], name(st))
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
        bond_row = posInLink!(leftmap,left=>lqn)
        bond_col = posInLink!(rightmap,mult(onsite,right)=>lqn)
        bond_coef = convert(ValType,coef(term))
        q_leftbond_coefs = get!(leftbond_coefs,lqn,MatElem{ValType}[])
        push!(q_leftbond_coefs,MatElem(bond_row,bond_col,bond_coef))
      end

      rqn = sqn+lqn
      A_row = bond_col
      A_col = posInLink!(next_rightmap,right=>rqn)
      site_coef = 1.0+0.0im
      if A_row == -1
        site_coef = coef(term)
      end
      if isempty(onsite)
        if isfermionic(right, sites)
          push!(onsite,SiteOp("F",n))
        else
          push!(onsite,SiteOp("Id",n))
        end
      end
      el = QNMatElem(lqn,rqn,A_row,A_col,MPOTerm(site_coef,onsite))
      push!(tempMPO[n],el)
    end
    rightmap = next_rightmap
    next_rightmap = Dict{Pair{OpTerm,QN},Int}()

    remove_dups!(tempMPO[n])

    if n > 1 && !isempty(leftbond_coefs)
      for (q,mat) in leftbond_coefs
        M = toMatrix(mat)
        U,S,V = svd(M)
        P = S.^2
        truncate!(P;maxdim=maxdim,cutoff=cutoff,mindim=mindim)
        tdim = length(P)
        nc = size(M,2)
        Vs[n][q] = Matrix{ValType}(V[1:nc,1:tdim])
      end
    end
  end

  #
  # Make MPO link indices
  #
  d0 = 2
  llinks = Vector{QNIndex}(undef,N+1)
  llinks[1] = Index(QN()=>d0;tags="Link,l=0")
  for n=1:N
    qi = Vector{Pair{QN,Int}}()
    if !haskey(Vs[n+1],QN())
      # Make sure QN=zero is first in list of sectors
      push!(qi,QN()=>d0)
    end
    for (q,Vq) in Vs[n+1]
      cols = size(Vq,2)
      if q==QN()
        # Make sure QN=zero is first in list of sectors
        insert!(qi,1,q=>d0+cols)
      else
        push!(qi,q=>cols)
      end
    end
    llinks[n+1] = Index(qi...;tags="Link,l=$n")
  end

  H = MPO(N)

  # Constants which define MPO start/end scheme
  startState = 2
  endState = 1

  for n=1:N
    finalMPO = Dict{Tuple{QN,OpTerm},Matrix{ValType}}()

    ll = llinks[n]
    rl = llinks[n+1]

    function defaultMat(ll,rl,lqn,rqn) 
      #ldim = qnblockdim(ll,lqn)
      #rdim = qnblockdim(rl,rqn)
      ldim = blockdim(ll, lqn)
      rdim = blockdim(rl, rqn)
      return zeros(ValType, ldim, rdim)
    end

    idTerm = [SiteOp("Id",n)]
    finalMPO[(QN(),idTerm)] = defaultMat(ll,rl,QN(),QN())
    idM = finalMPO[(QN(),idTerm)]
    idM[1,1] = 1.0
    idM[2,2] = 1.0

    for el in tempMPO[n]
      t = el.val
      (abs(coef(t)) > eps()) || continue
      A_row = el.row
      A_col = el.col

      M = get!(finalMPO,(el.rowqn,ops(t)),defaultMat(ll,rl,el.rowqn,el.colqn))

      # rowShift and colShift account for
      # special entries in the zero-QN sector
      # of the MPO
      rowShift = (el.rowqn == QN()) ? 2 : 0
      colShift = (el.colqn == QN()) ? 2 : 0

      ct = convert(ValType,coef(t))
      if A_row==-1 && A_col==-1 #onsite term
        M[startState,endState] += ct
      elseif A_row==-1 #term starting on site n
        VR = Vs[n+1][el.colqn]
        for c=1:size(VR,2)
          z = ct*VR[A_col,c]
          M[startState,colShift+c] += z
        end
      elseif A_col==-1 #term ending on site n
        VL = Vs[n][el.rowqn]
        for r=1:size(VL,2)
          z = ct*conj(VL[A_row,r])
          M[rowShift+r,endState] += z
        end
      else
        VL = Vs[n][el.rowqn]
        VR = Vs[n+1][el.colqn]
        for r=1:size(VL,2),c=1:size(VR,2)
          z = ct*conj(VL[A_row,r])*VR[A_col,c]
          M[rowShift+r,colShift+c] += z
        end
      end
    end

    s = sites[n]
    #H[n] = emptyITensor(dag(s),s',dag(ll),rl)
    H[n] = emptyITensor()
    for (q_op,M) in finalMPO
      op_prod = q_op[2]
      Op = computeSiteProd(sites,op_prod)

      rq = q_op[1]
      sq = flux(Op)
      cq = rq-sq

      #rn = qnblocknum(ll,rq)
      #cn = qnblocknum(rl,cq)
      rn = block(first, ll, rq)
      cn = block(first, rl, cq)

      #TODO: wrap following 3 lines into a function
      _block = Block(rn, cn)
      T = BlockSparseTensor(ValType,[_block],IndexSet(dag(ll),rl))
      #blockview(T, _block) .= M
      T[_block] .= M

      IT = itensor(T)
      H[n] += IT * Op
    end

  end

  L = emptyITensor(llinks[1])
  L[startState] = 1.0

  R = emptyITensor(dag(llinks[N+1]))
  R[endState] = 1.0

  H[1] *= L
  H[N] *= R

  return H
end #qn_svdMPO

function sorteachterm!(ampo::AutoMPO, sites)
  ampo = copy(ampo)
  isless_site(o1::SiteOp, o2::SiteOp) = site(o1) < site(o2)
  N = length(sites)
  for t in data(ampo)
    Nt = length(t.ops)
    prevsite = N+1 #keep track of whether we are switching
                   #to a new site to make sure F string
                   #is only placed at most once for each site

    # Sort operators in t by site order,
    # and keep the permutation used, perm, for analysis below
    perm = Vector{Int}(undef,Nt)
    sortperm!(perm,t.ops, alg=InsertionSort, lt=isless_site)

    t.ops = t.ops[perm]

    # Identify fermionic operators,
    # zeroing perm for bosonic operators,
    # and inserting string "F" operators
    rhs_parity = +1
    for n=Nt:-1:1
      currsite = site(t.ops[n])
      fermionic = has_fermion_string(name(t.ops[n]),
                                     sites[site(t.ops[n])])
      if (rhs_parity==-1) && (currsite < prevsite)
        # Put local piece of Jordan-Wigner string emanating
        # from fermionic operators to the right
        # (Remaining F operators will be put in by svdMPO)
        t.ops[n] = SiteOp("$(name(t.ops[n]))*F",site(t.ops[n]))
      end
      prevsite = currsite

      if fermionic
        rhs_parity = -rhs_parity
      else
        # Ignore bosonic operators in perm
        # by zeroing corresponding entries
        perm[n] = 0
      end
    end
    if rhs_parity == -1
      error("Total parity-odd fermionic terms not yet supported by AutoMPO")
    end
    # Keep only fermionic op positions (non-zero entries)
    filter!(!iszero,perm)
    # Account for anti-commuting, fermionic operators 
    # during above sort; put resulting sign into coef

    t.coef *= parity_sign(perm)
  end
  return ampo
end

function sortmergeterms!(ampo::AutoMPO)

  sort!(data(ampo))

  # Merge (add) terms with same operators
  da = data(ampo)
  ndata = MPOTerm[]
  last_term = copy(da[1])
  for n=2:length(da)
    if ops(da[n])==ops(last_term)
      last_term.coef += coef(da[n])
    else
      push!(ndata,last_term)
      last_term = copy(da[n])
    end
  end
  push!(ndata,last_term)

  setdata!(ampo,ndata)
  return ampo
end

"""
    MPO(ampo::AutoMPO,sites::Vector{<:Index};kwargs...)
       
Convert an AutoMPO object `ampo` to an
MPO, with indices given by `sites`. The
resulting MPO will have the indices
`sites[1], sites[1]', sites[2], sites[2]'`
etc. The conversion is done by an algorithm
that compresses the MPO resulting from adding
the AutoMPO terms together, often achieving
the minimum possible bond dimension.

# Examples
```julia
ampo = AutoMPO()
ampo += ("Sz",1,"Sz",2)
ampo += ("Sz",2,"Sz",3)
ampo += ("Sz",3,"Sz",4)

sites = siteinds("S=1/2",4)
H = MPO(ampo,sites)
```
"""
function MPO(ampo::AutoMPO,
             sites::Vector{<:Index};
             kwargs...)::MPO
  length(data(ampo)) == 0 && error("AutoMPO has no terms")

  sorteachterm!(ampo,sites)
  sortmergeterms!(ampo)

  if hasqns(sites[1])
    return qn_svdMPO(ampo,sites;kwargs...)
  end
  return svdMPO(ampo,sites;kwargs...)
end

