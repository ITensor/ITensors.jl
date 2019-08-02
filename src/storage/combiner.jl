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
        c_pos = indexpositions(dis, cind)[1]
        Nis   = IndexSet(length(dis) + length(Cis) - 2)
        i = 1
        for j in 1:length(dis)
            if j == c_pos
                for k in 2:length(Cis)
                    Nis[i] = Cis[k]
                    i += 1
                end
            else
                Nis[i] = dis[j]
                i += 1
            end
        end
        return Nis, Dstore
    else # lacks combined index, combine
        # dis doesn't have cind, replace
        # Cis[1], Cis[2], ... with cind, may need to permute
        j_1_pos = indexpositions(dis, Cis[2])[1]
        j_1_pos == 0 && throw(ArgumentError("no contracted indices in combiner-tensor product"))
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
        Nis = IndexSet(length(dis) + 2 - length(Cis))
        if contig_sameord
            # this is nasty
            i = 1
            for j in 1:j_1_pos-1
                Nis[i] = dis[j]
                i += 1
            end
            Nis[i] = cind
            i += 1
            for j in (j_1_pos + length(Cis) - 1):length(dis)
                Nis[i] = dis[j]
                i += 1
            end
        else # permutation required
            # Set P destination values to -1 to mark indices that need to be assigned destinations:
            P = fill(-1, length(dis))
            # permute combined indices to the front, in same order as in Cis:
            ni = 1
            for c in 2:length(Cis)
                j = indexpositions(dis, Cis[c])[1]
                j == 0 && throw(ArgumentError("combiner missing index"))
                P[j] = ni
                ni += 1
            end
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
            Dstore = Dense{eltype(Dstore)}(vec(permutedims(reshape(data(Dstore), dims(dis)), invperm(P))))
        end
        return Nis, Dstore
    end
end

storage_contract(Dstore::Dense, dis::IndexSet, CSstore::CombinerStorage, Cis::IndexSet) = storage_contract(CSstore, Cis, Dstore, dis)
