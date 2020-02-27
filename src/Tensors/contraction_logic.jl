
const Labels{N} = NTuple{N,Int}

# Automatically determine the output labels given
# input labels of a contraction
function contract_labels(T1labels::Labels{N1},
                         T2labels::Labels{N2}) where {N1,N2}
  ncont = 0
  for i in T1labels
    i < 0 && (ncont += 1)
  end
  NR = N1+N2-2*ncont
  Rlabels = Vector{Int}(undef,NR)
  u = 1
  # TODO: use Rlabels, don't assume ncon convention
  @inbounds for i ∈ 1:N1
    if(T1labels[i] > 0)
      Rlabels[u] = T1labels[i];
      u += 1
    end
  end
  @inbounds for i ∈ 1:N2
    if(T2labels[i] > 0)
      Rlabels[u] = T2labels[i];
      u += 1
    end
  end
  return Labels{NR}(Rlabels)
end

function contract_inds(T1is,
                       T1labels::Labels{N1},
                       T2is,
                       T2labels::Labels{N2},
                       Rlabels::Labels{NR}) where {N1,N2,NR}
  ncont = 0
  for i in T1labels
    i < 0 && (ncont += 1)
  end
  IndT1 = eltype(T1is)
  Ris = Vector{IndT1}(undef,NR)
  u = 1
  # TODO: use Rlabels, don't assume ncon convention
  for i1 ∈ 1:N1
    if T1labels[i1] > 0
      Ris[u] = T1is[i1]; 
      u += 1 
    else
      # This is to check that T1is and T2is
      # can contract
      i2 = findfirst(==(T1labels[i1]),T2labels)
      dir(T1is[i1]) == -dir(T2is[i2]) || error("Indices must have opposite directions to contract")
    end
  end
  for i2 ∈ 1:N2
    if T2labels[i2] > 0
      Ris[u] = T2is[i2]; 
      u += 1 
    end
  end
  IndsT1 = typeof(T1is)
  IndsR = similar_type(IndsT1,Val{NR})
  return IndsR(tuple(Ris...))
end

mutable struct ContractionProperties{NA,NB,NC}
  ai::NTuple{NA,Int}
  bi::NTuple{NB,Int}
  ci::NTuple{NC,Int}
  nactiveA::Int 
  nactiveB::Int 
  nactiveC::Int
  AtoB::MVector{NA,Int}
  AtoC::MVector{NA,Int}
  BtoC::MVector{NB,Int}
  permuteA::Bool
  permuteB::Bool
  permuteC::Bool
  dleft::Int 
  dmid::Int
  dright::Int
  ncont::Int
  Acstart::Int
  Bcstart::Int
  Austart::Int
  Bustart::Int
  PA::Vector{Int}
  PB::Vector{Int}
  PC::Vector{Int}
  ctrans::Bool
  newArange::Vector{Int}
  newBrange::Vector{Int}
  newCrange::Vector{Int}
  function ContractionProperties(ai::NTuple{NA,Int},
                                 bi::NTuple{NB,Int},
                                 ci::NTuple{NC,Int}) where {NA,NB,NC}
    new{NA,NB,NC}(ai,bi,ci,0,0,0,
                  ntuple(_->0,NA),ntuple(_->0,NA),ntuple(_->0,NB),
                  false,false,false,1,1,1,0,
                  NA,NB,NA,NB,
                  Vector{Int}(),Vector{Int}(),Vector{Int}(),
                  false,
                  Vector{Int}(),Vector{Int}(),Vector{Int}())
  end
end

function compute_perms!(props::ContractionProperties{NA,NB,NC}) where 
                                                           {NA,NB,NC}
  #length(props.AtoB)!=0 && return

  #props.AtoB = fill(0,NA)
  #props.AtoC = fill(0,NA)
  #props.BtoC = fill(0,NB)
  for i = 1:NA
    for j = 1:NB
      if props.ai[i]==props.bi[j]
        props.ncont += 1
        #TODO: check this if this should be i,j or i-1,j-1 (0-index or 1-index)
        i<=props.Acstart && (props.Acstart = i)
        j<=props.Bcstart && (props.Bcstart = j)
        props.AtoB[i] = j
        break
      end
    end
  end

  for i = 1:NA
    for k = 1:NC
      if props.ai[i]==props.ci[k]
        #TODO: check this if this should be i,j or i-1,j-1 (0-index or 1-index)
        i<=props.Austart && (props.Austart = i)
        props.AtoC[i] = k
        break
      end
    end
  end

  for j = 1:NB
    for k = 1:NC
      if props.bi[j]==props.ci[k]
        #TODO: check this if this should be i,j or i-1,j-1 (0-index or 1-index)
        j<=props.Bustart && (props.Bustart = j)
        props.BtoC[j] = k
        break
      end
    end
  end

end

function checkACsameord(props::ContractionProperties)::Bool
  props.Austart>=length(props.ai) && return true
  aCind = props.AtoC[props.Austart]
  for i = 1:length(props.ai)
    if !contractedA(props,i)
      props.AtoC[i]!=aCind && return false
      aCind += 1
    end
  end
  return true
end

function checkBCsameord(props::ContractionProperties)::Bool
  props.Bustart>=length(props.bi) && return true
  bCind = props.BtoC[props.Bustart]
  for i = 1:length(props.bi)
    if !contractedB(props,i)
      props.BtoC[i]!=bCind && return false
      bCind += 1
    end
  end
  return true
end

contractedA(props::ContractionProperties,i::Int) = (props.AtoC[i]<1)
contractedB(props::ContractionProperties,i::Int) = (props.BtoC[i]<1)
Atrans(props::ContractionProperties) = contractedA(props,1)
Btrans(props::ContractionProperties) = !contractedB(props,1)
Ctrans(props::ContractionProperties) = props.ctrans

function compute_contraction_properties!(props::ContractionProperties{NA,NB,NC},
                                         A,B,C) where {NA,NB,NC}
  compute_perms!(props)

  #Use props.PC.size() as a check to see if we've already run this
  length(props.PC)!=0 && return

  ra = NA #length(props.ai)
  rb = NB #length(props.bi)
  rc = NC #length(props.ci)

  props.PC = fill(0,rc)

  props.dleft = 1
  props.dmid = 1
  props.dright = 1
  c = 1
  for i = 1:ra
    if !contractedA(props,i)
      props.dleft *= size(A,i)
      props.PC[props.AtoC[i]] = c
      c += 1
    else
      props.dmid *= size(A,i)
    end
  end
  for j = 1:rb
    if !contractedB(props,j)
      props.dright *= size(B,j)
      props.PC[props.BtoC[j]] = c
      c += 1
    end
  end

  if !is_trivial_permutation(props.PC)
    props.permuteC = true
    if checkBCsameord(props) && checkACsameord(props)
      #Can avoid permuting C by 
      #computing Bt*At = Ct
      props.ctrans = true
      props.permuteC = false
    end
  end

  #Check if A can be treated as a matrix without permuting
  props.permuteA = false
  if !(contractedA(props,1) || contractedA(props,ra))
    #If contracted indices are not all at front or back, 
    #will have to permute A 
    props.permuteA = true
  else
    #Contracted ind start at front or back, check if contiguous
    #TODO: check that the limits are correct (1-indexed vs. 0-indexed)
    for i = 1:props.ncont
      if !contractedA(props,props.Acstart+i-1)
        #Contracted indices not contiguous, must permute
        props.permuteA = true
        break
      end
    end
  end

  #Check if B is matrix-like
  props.permuteB = false
  if !(contractedB(props,1) || contractedB(props,rb))
    #If contracted indices are not all at front or back, 
    #will have to permute B
    props.permuteB = true
  else
    #TODO: check that the limits are correct (1-indexed vs. 0-indexed)
    for i = 1:props.ncont
      if !contractedB(props,props.Bcstart+i-1)
        #Contracted inds not contiguous, permute
        props.permuteB = true
        break
      end
    end
  end

  if !props.permuteA && !props.permuteB
    #Check if contracted inds. in same order
    #TODO: check these limits are correct
    for i = 1:props.ncont
      if props.AtoB[props.Acstart+i-1]!=(props.Bcstart+i-1)
        #If not in same order, 
        #must permute one of A or B
        #so permute the smaller one
        props.dleft<props.dright ? (props.permuteA = true) : (props.permuteB = true)
        break
      end
    end
  end

  if props.permuteC && !(props.permuteA && props.permuteB)
    PCost(d::Real) = d*d
    #Could avoid permuting C if
    #permute both A and B, worth it?
    pCcost = PCost(props.dleft*props.dright)
    extra_pABcost = 0
    !props.permuteA && (extra_pABcost += PCost(props.dleft*props.dmid))
    !props.permuteB && (extra_pABcost += PCost(props.dmid*props.dright))
    if extra_pABcost<pCcost
      props.permuteA = true
      props.permuteB = true
      props.permuteC = false
    end
  end

  if props.permuteA
    props.PA = fill(0,ra)
    #Permute contracted indices to the front,
    #in the same order as on B
    newi = 0
    #TODO: check this is correct for 1-indexing
    bind = props.Bcstart
    for i = 1:props.ncont
      while !contractedB(props,bind) bind += 1 end
      j = findfirst(==(props.bi[bind]),props.ai)
      newi += 1
      props.PA[newi] = j
      bind += 1
    end
    #Reset p.AtoC:
    fill!(props.AtoC,0)
    #Permute uncontracted indices to
    #appear in same order as on C
    #TODO: check this is correct for 1-indexing
    for k = 1:rc
      j = findfirst(==(props.ci[k]),props.ai)
      if !isnothing(j)
        props.AtoC[newi] = k
        props.PA[newi+1] = j
        newi += 1
      end
      newi==ra && break
    end
  end

  ##Also update props.Austart,props.Acstart
  props.Acstart = ra+1
  props.Austart = ra+1
  #TODO: check this is correct for 1-indexing
  for i = 1:ra
    if contractedA(props,i)
      props.Acstart = min(i,props.Acstart)
    else
      props.Austart = min(i,props.Austart)
    end
    #props.newArange = permute_extents([size(A)...],props.PA)
    props.newArange = [size(A)...][props.PA]
  end

  if(props.permuteB)
    props.PB = fill(0,rb)
    #TODO: check this is correct for 1-indexing
    newi = 0 #1
    if(props.permuteA)
      #A's contracted indices already set to
      #be in same order as B above, so just
      #permute contracted indices to the front
      #keeping relative order
      #TODO: how to translate this for loop?
      #for(int i = props.Bcstart; newi < props.ncont; ++newi)
      i = props.Bcstart
      #TODO: check this is correct for 1-indexing
      while newi < props.ncont
        while !contractedB(props,i) i += 1 end
        props.PB[newi+1] = i
        i += 1
        newi += 1
      end
    else
      #Permute contracted indices to the
      #front and in same order as on A
      aind = props.Acstart
      for i = 0:(props.ncont-1)
        while !contractedA(props,aind) aind += 1 end
        j = findfirst(==(props.ai[aind]),props.bi)
        newi += 1
        props.PB[newi] = j
        aind += 1
      end
    end
    #Reset p.BtoC:
    fill!(props.BtoC,0)
    #Permute uncontracted indices to
    #appear in same order as on C
    for k = 1:rc
      j = findfirst(==(props.ci[k]),props.bi)
      if !isnothing(j)
        props.BtoC[newi] = k
        props.PB[newi+1] = j
        newi += 1
      end
      newi==rb && break
    end
    props.Bcstart = rb
    props.Bustart = rb
    for i = 1:rb
      if(contractedB(props,i))
          props.Bcstart = min(i,props.Bcstart)
      else
          props.Bustart = min(i,props.Bustart)
      end
    end
    #props.newBrange = permute_extents([size(B)...],props.PB)
    props.newBrange = [size(B)...][props.PB]
  end

  if props.permuteA || props.permuteB
    #Recompute props.PC
    c = 1
    #TODO: check this is correct for 1-indexing
    for i = 1:ra
      if !contractedA(props,i)
        props.PC[props.AtoC[i]] = c
        c += 1
      end
    end
    #TODO: check this is correct for 1-indexing
    for j = 1:rb
      if !contractedB(props,j)
        props.PC[props.BtoC[j]] = c
        c += 1
      end
    end
    props.ctrans = false
    if(is_trivial_permutation(props.PC))
      props.permuteC = false
    else
      props.permuteC = true
      #Here we already know since pc_triv = false that
      #at best indices from B precede those from A (on result C)
      #so if both sets remain in same order on C 
      #just need to transpose C, not permute it
      if  checkBCsameord(props) && checkACsameord(props)
        props.ctrans = true
        props.permuteC = false
      end
    end
  end

  if props.permuteC
    Rb = Int[]
    if !props.permuteA
      #TODO: check this is correct for 1-indexing
      for i = 1:ra
        if !contractedA(props,i)
          push!(Rb,size(A,i))
        end
      end
    else
      #TODO: check this is correct for 1-indexing
      for i = 1:ra
        if !contractedA(props,i)
          push!(Rb,size(props.newArange,i))
        end
      end
    end
    if !props.permuteB
      #TODO: check this is correct for 1-indexing
      for j = 1:rb
        if !contractedB(props,j)
          push!(Rb,size(B,j))
        end
      end
    else
      #TODO: check this is correct for 1-indexing
      for j = 1:rb
        if !contractedB(props,j)
          push!(Rb,size(props.newBrange,j))
        end
      end
    end
    props.newCrange = Rb
  end

end

