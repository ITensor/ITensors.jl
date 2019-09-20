
struct CombinerStorage <: TensorStorage
  ci::Index
end

function storage_contract(CSstore::CombinerStorage,
                          Cis::IndexSet,
                          Dstore::Dense,
                          dis::IndexSet)
  cind = Cis[1]
  if hasindex(dis, cind) # has combined index, uncombine
    cpos = findindex(dis,cind)
    dinds = inds(dis)
    Cinds = inds(Cis)
    Nis = IndexSet(vcat(dinds[1:cpos-1], Cinds[2:end], dinds[cpos+1:end]))
    return Nis, Dstore
  else # lacks combined index, combine
    # dis doesn't have cind, replace
    # Cis[1], Cis[2], ... with cind, may need to permute
    j_1_pos = findindex(dis,Cis[2])
    if j_1_pos < 1 
      throw(ArgumentError("tensor missing index $(Cis[2]) in combiner-tensor product"))
    end

    # Check if Cis[2], Cis[3], ... are grouped together (contiguous)
    # and in same order as on combiner
    contig_sameord = true
    j = j_1_pos+1
    for c=3:length(Cis)
      if j > length(dis) || (dis[j] != Cis[c])
        contig_sameord = false
        break
      end
      j += 1
    end

    if contig_sameord
      lend = j_1_pos-1
      rstart = j_1_pos+length(Cis)-1
      dinds = inds(dis)
      Nis = IndexSet(vcat(dinds[1:lend], cind, dinds[rstart:end]))
    else # permutation required
      # Set P destination values to -1 to mark indices that need to be assigned destinations:
      P = fill(-1, length(dis))
      # permute combined indices to the front, in same order as in Cis:
      ni = 1
      for c in 2:length(Cis)
        j = findindex(dis, Cis[c])
        if j < 1 
          throw(ArgumentError("tensor missing index $(Cis[c]) in combiner-tensor product"))
        end
        P[j] = ni
        ni += 1
      end

      Nis = IndexSet(length(dis)+2-length(Cis))
      Nis[1] = cind
      i = 2
      for j in 1:length(dis)
        if P[j] == -1
          P[j] = ni
          ni += 1
          Nis[i] = dis[j]
          i += 1
        end
      end

      ddata = vec(permutedims(reshape(data(Dstore), dims(dis)), invperm(P)))
      Dstore = Dense{eltype(Dstore)}(ddata)
    end
    return Nis, Dstore
  end
end

function storage_contract(Dstore::Dense, 
                          dis::IndexSet, 
                          CSstore::CombinerStorage, 
                          Cis::IndexSet) 
  return storage_contract(CSstore, Cis, Dstore, dis)
end
