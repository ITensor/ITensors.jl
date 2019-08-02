export Combiner

struct CombinerStorage <: TensorStorage
  ci::Index
end

# use eachindex here
function storage_contract(CSstore::CombinerStorage,
                          Cis::IndexSet,
                          Dstore::Dense,
                          dis::IndexSet)
    cind = Cis[1]
    if hasindex(dis, cind) # has combined index, uncombine
        cpos = findindex(dis,cind)
        ninds = vcat(inds(dis)[1:cpos-1],inds(Cis)[2:end],inds(dis)[cpos+1:end])
        Nis = IndexSet(ninds)
        return Nis, Dstore
    else # lacks combined index, combine
        # dis doesn't have cind, replace
        # Cis[1], Cis[2], ... with cind, may need to permute
        j_1_pos = indexpositions(dis, Cis[2])[1]
        j_1_pos < 1 && throw(ArgumentError("no contracted indices in combiner-tensor product"))
        # Check if Cis[2], Cis[3], ... are grouped together (contiguous)
        # and in same order as on combiner
        c_pos = indexpositions(Cis, cind)[1]
        extent = min(length(dis) - j_1_pos - 1, length(Cis) - c_pos - 1)
        contig_sameord = true
        c = 2
        for j in j_1_pos : length(dis)
            contig_sameord &= dis[j] == Cis[c] 
            !contig_sameord && break
            c += 1
            c > length(Cis) && break
        end
        contig_sameord &= c == length(Cis) + 1
        if contig_sameord
            lend = j_1_pos-1
            rstart = j_1_pos+length(Cis)-1
            ninds = vcat(inds(dis)[1:lend],cind,inds(dis)[rstart:end])
            Nis = IndexSet(ninds)
        else # permutation required

            # Set P destination values to -1 to mark indices that need to be assigned destinations:
            P = fill(-1, length(dis))
            # permute combined indices to the front, in same order as in Cis:
            ni = 1
            for c in 2:length(Cis)
              j = findindex(dis, Cis[c])
              j < 1 && throw(ArgumentError("combiner missing index"))
              P[j] = ni
              ni += 1
            end

            Nis = IndexSet(length(dis)+2-length(Cis))
            Nis[1] = cind
            i = 2
            for j in 1:length(dis)
                if P[j] == -1
                    P[j] = ni
                    Nis[i] = dis[j]
                    i += 1
                    ni += 1
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
