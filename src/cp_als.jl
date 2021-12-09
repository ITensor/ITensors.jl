export cp_als

## The Alternating Least-Squares (ALS) algorithm for CP-decomposition with ITensors.jl
# Based on "Tensor Decompositions and Applications", Tamara G. Kolda and Brett W. Bader. SIAM Review 51(3), 2009
# Porting the code from https://www.kaggle.com/nicw102168/rank-of-random-2x2x2-tensors

using ITensors
using ITensors.LinearAlgebra


function cp_als(X, R; maxiter=1000, test_period=1, tol=1e-10)
    # initialize with spheric-random colunms
    A = map([randomITensor(Iₙ, Index(R)) for Iₙ in inds(X)]) do An
        aλ = columnnorms(An)
        rr = Index(R)
        iλ = diagITensor(aλ.^-1, rr, inds(An)[2])
        replaceind(An, inds(An)[2], rr) * iλ
    end
    cp_als_(X, R, A, maxiter=maxiter, test_period=test_period, tol=tol)
end

function cp_als_(X, R, A; maxiter=1000, test_period=1, tol=1e-10)
    N = order(X)

    rind = [inds(An)[2] for An in A]
    λ = nothing
    iteration = nothing

    for it in 1:maxiter
        iteration=it

        for n in 1:N
            V = reduce((a,b)->a.*b, array(A[m])' * array(A[m]) for m in 1:N if m != n) ## is there a better, ITensor way to do that?
            Vinv = ITensors.LinearAlgebra.pinv(V)

            W = reduce(khatrirao, [A[m] for m in N:-1:1 if m != n])
            Xn = unfold(X,n)
            qq = diagITensor(ones(dim(inds(Xn)[2])), inds(Xn)[2], inds(W)[1])

            rr = Index(R)
            newAn = Xn * qq * W * itensor(Vinv, inds(W)[2], rr)

            aλ = columnnorms(newAn)
            λ = diagITensor(aλ, rind...)
            iλ = diagITensor(aλ.^-1, rr, inds(A[n])[2])

            A[n] = newAn * iλ
        end

        if it % test_period==0
            X̂ = prod(A) * λ  # calculate PARAFAC model

            if norm(X - X̂) < tol
                break
            end
        end
    end

    return λ, A, iteration
end

"""Mode-n matricization of tensor X, or X₍ₙ₎ in Kolda & Bader 2009."""
function unfold(X, n)
    matn = combiner(uniqueinds(X, IndexSet(inds(X)[n])))
    X*matn
end

"""The Khatri-Rao product, or A ⊙ B."""
function khatrirao(A, B)
    K = dim(inds(A)[2])
    AB = A * B * diagITensor(ones(K), inds(A)[2], inds(B)[2], Index(K))
    matk = combiner(inds(B)[1], inds(A)[1])
    matk * AB
end

function columnnorms(A)
    slicenorms(A, inds(A)[2])
end

function slicenorm(A::ITensor, i::Index, val::Int)
  return norm(A * onehot(i => val))
end
slicenorms(A::ITensor, i::Index) = [slicenorm(A, i, n) for n in 1:dim(i)]
