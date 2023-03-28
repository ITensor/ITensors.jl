const Fu=F_utilities

function build_Ω(N::Int)
    return Fu.Build_Omega(div(N,2))
end



function build_Fxpxx(N::Int)
    return Fu.Build_FxpTxx(div(N,2))
end

function get_ordered_Schur(h::AbstractMatrix)
    T,O,vals=order_schur(schur(h))
    
end

function compare_schur(h)
    T,O,vals=get_ordered_Schur(h)
    T2,O2=Fu.Diag_real_skew(h)
    @show T
    @show T2
    @show O
    @show O2
end

function order_schur(F::Schur)
    T=F.Schur
    O=F.vectors #column vectors are Schur vectors
    
    N=size(T,1)
    n=div(N,2)
    shuffled_inds=Vector{Int}[]
    ElT=eltype(T)
    vals=ElT[]
    ###build a permutation matrix that takes care of the ordering
    ###build block local rotation first
    ###then permute blocks for overall ordering
    for i in 1:n
        ind=2*i-1
        val=T[ind,ind+1]
        if val>=0
            push!(shuffled_inds,[ind,ind+1])
        else
            push!(shuffled_inds,[ind+1,ind])
        end
        push!(vals,abs(val))
    end
    perm=sortperm(vals,rev=true)    ##we want the upper left corner to be the largest absolute value eigval pair?
    sort!(vals)
    shuffled_inds=reduce(vcat,shuffled_inds[perm])

    T=T[shuffled_inds,shuffled_inds]
    O=O[:,shuffled_inds]
    return T,O,vals ##vals being only positive, and of length n and not N
end

"""
function Build_Omega(N_f)
    #Build the matrix omega of dimension 2*N_f, that is for N_f fermions.
     Omega                                 = zeros(Complex{Float64}, 2*N_f, 2*N_f);
     Omega[1:N_f,1:N_f]                    = (1/(sqrt(2)))*eye(N_f);
     Omega[1:N_f,(1:N_f)+N_f]              = (1/(sqrt(2)))*eye(N_f);
     Omega[(1:N_f)+N_f,1:N_f]              = im*(1/(sqrt(2)))*eye(N_f);
     Omega[(1:N_f)+N_f,(1:N_f)+N_f]        = -im*(1/(sqrt(2)))*eye(N_f);
     return Omega
    end
    
function Build_FxxTxp(N_f)
    FxxTxp = zeros(Int64, 2*N_f, 2*N_f);
    for iiter=1:N_f
    FxxTxp[2*iiter-1,iiter]    = 1;
    FxxTxp[2*iiter, iiter+N_f] = 1;
    end
    return FxxTxp
end

function Build_FxpTxx(N_f)
    FxpTxx = zeros(Int64, 2*N_f, 2*N_f)
    for iiter=1:N_f
    FxpTxx[iiter,2*iiter-1]    = 1;
    FxpTxx[iiter+N_f, 2*iiter] = 1;
    end

    return FxpTxx
end
"""