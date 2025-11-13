using NDTensors.RankFactorization: Spectrum

"""
    TruncSVD

ITensor factorization type for a truncated singular-value
decomposition, returned by `svd`.
"""
struct TruncSVD
    U::ITensor
    S::ITensor
    V::ITensor
    spec::Spectrum
    u::Index
    v::Index
end

# iteration for destructuring into components `U,S,V,spec,u,v = S`
iterate(S::TruncSVD) = (S.U, Val(:S))
iterate(S::TruncSVD, ::Val{:S}) = (S.S, Val(:V))
iterate(S::TruncSVD, ::Val{:V}) = (S.V, Val(:spec))
iterate(S::TruncSVD, ::Val{:spec}) = (S.spec, Val(:u))
iterate(S::TruncSVD, ::Val{:u}) = (S.u, Val(:v))
iterate(S::TruncSVD, ::Val{:v}) = (S.v, Val(:done))
iterate(S::TruncSVD, ::Val{:done}) = nothing

@doc """
    svd(A::ITensor, inds::Index...; <keyword arguments>)

Singular value decomposition (SVD) of an ITensor `A`, computed
by treating the "left indices" provided collectively
as a row index, and the remaining "right indices" as a
column index (matricization of a tensor).

The first three return arguments are `U`, `S`, and `V`, such that
`A ≈ U * S * V`.

Whether or not the SVD performs a trunction depends on the keyword
arguments provided.

If the left or right set of indices are empty, all input indices are
put on `V` or `U` respectively. To specify an empty set of left indices,
you must explicitly use `svd(A, ())` (`svd(A)` is currently undefined).

# Examples

Computing the SVD of an order-three ITensor, such that the indices
i and k end up on U and j ends up on V

```
i = Index(2)
j = Index(5)
k = Index(2)
A = random_itensor(i, j, k)
U, S, V = svd(A, i, k);
@show norm(A - U * S * V) <= 10 * eps() * norm(A)
```

The following code will truncate the last 2 singular values,
since the total number of singular values is 4.
The norm of the difference with the original tensor
will be the sqrt root of the sum of the squares of the
singular values that get truncated.

```
trunc, Strunc, Vtrunc = svd(A, i, k; maxdim=2);
@show norm(A - Utrunc * Strunc * Vtrunc) ≈ sqrt(S[3, 3]^2 + S[4, 4]^2)
```

Alternatively we can specify that we want to truncate
the weights of the singular values up to a certain cutoff,
so the total error will be no larger than the cutoff.

```
Utrunc2, Strunc2, Vtrunc2 = svd(A, i, k; cutoff=1e-10);
@show norm(A - Utrunc2 * Strunc2 * Vtrunc2) <= 1e-10
```

# Keywords
- `maxdim::Int`: the maximum number of singular values to keep.
- `mindim::Int`: the minimum number of singular values to keep.
- `cutoff::Float64`: set the desired truncation error of the SVD,
   by default defined as the sum of the squares of the smallest singular values.
- `lefttags::String = "Link,u"`: set the tags of the Index shared by `U` and `S`.
- `righttags::String = "Link,v"`: set the tags of the Index shared by `S` and `V`.
- `alg::String = "divide_and_conquer"`. Options:
- `"divide_and_conquer"` - A divide-and-conquer algorithm.
     LAPACK's gesdd. Fast, but may lead to some innacurate singular values
     for very ill-conditioned matrices. Also may sometimes fail to converge,
     leading to errors (in which case "qr_iteration" or "recursive" can be tried).
  - `"qr_iteration"` - Typically slower but more accurate for very
     ill-conditioned matrices compared to `"divide_and_conquer"`.
     LAPACK's gesvd.
  - `"recursive"` - ITensor's custom svd. Very reliable, but may be slow if
     high precision is needed. To get an `svd` of a matrix `A`, an
     eigendecomposition of ``A^{\\dagger} A`` is used to compute `U` and then
     a `qr` of ``A^{\\dagger} U`` is used to compute `V`. This is performed
     recursively to compute small singular values.
- `use_absolute_cutoff::Bool = false`: set if all probability weights below
   the `cutoff` value should be discarded, rather than the sum of discarded
   weights.
- `use_relative_cutoff::Bool = true`: set if the singular values should be
   normalized for the sake of truncation.
- `min_blockdim::Int = 0`: for SVD of block-sparse or QN ITensors, require
   that the number of singular values kept be greater than or equal to
   this value when possible

See also: [`factorize`](@ref), [`eigen`](@ref)
"""
function svd(
        A::ITensor,
        Linds...;
        leftdir = nothing,
        rightdir = nothing,
        lefttags = nothing,
        righttags = nothing,
        mindim = nothing,
        maxdim = nothing,
        cutoff = nothing,
        alg = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = nothing,
        min_blockdim = nothing,
        # Deprecated
        utags = lefttags,
        vtags = righttags,
    )
    lefttags = NDTensors.replace_nothing(lefttags, ts"Link,u")
    righttags = NDTensors.replace_nothing(righttags, ts"Link,v")

    # Deprecated
    utags = NDTensors.replace_nothing(utags, ts"Link,u")
    vtags = NDTensors.replace_nothing(vtags, ts"Link,v")

    Lis = commoninds(A, indices(Linds...))
    Ris = uniqueinds(A, Lis)

    Lis_original = Lis
    Ris_original = Ris
    if isempty(Lis_original)
        α = trivial_index(Ris)
        vLα = onehot(datatype(A), α => 1)
        A *= vLα
        Lis = [α]
    end
    if isempty(Ris_original)
        α = trivial_index(Lis)
        vRα = onehot(datatype(A), α => 1)
        A *= vRα
        Ris = [α]
    end

    CL = combiner(Lis...; dir = leftdir)
    CR = combiner(Ris...; dir = rightdir)
    AC = A * CR * CL
    cL = combinedind(CL)
    cR = combinedind(CR)
    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end

    USVT = svd(
        tensor(AC);
        mindim,
        maxdim,
        cutoff,
        alg,
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
    )
    if isnothing(USVT)
        return nothing
    end
    UT, ST, VT, spec = USVT
    UC, S, VC = itensor(UT), itensor(ST), itensor(VT)

    u = commonind(S, UC)
    v = commonind(S, VC)

    U = UC * dag(CL)
    V = VC * dag(CR)

    U = settags(U, utags, u)
    S = settags(S, utags, u)
    S = settags(S, vtags, v)
    V = settags(V, vtags, v)

    u = settags(u, utags)
    v = settags(v, vtags)

    if isempty(Lis_original)
        U *= dag(vLα)
    end
    if isempty(Ris_original)
        V *= dag(vRα)
    end

    return TruncSVD(U, S, V, spec, u, v)
end

svd(A::ITensor; kwargs...) = error("Must specify indices in `svd`")

"""
    TruncEigen

ITensor factorization type for a truncated eigenvalue
decomposition, returned by `eigen`.
"""
struct TruncEigen
    D::ITensor
    V::ITensor
    Vt::ITensor
    spec::Spectrum
    l::Index
    r::Index
end

# iteration for destructuring into components `D, V, spec, l, r = E`
iterate(E::TruncEigen) = (E.D, Val(:V))
iterate(E::TruncEigen, ::Val{:V}) = (E.V, Val(:spec))
iterate(E::TruncEigen, ::Val{:spec}) = (E.spec, Val(:l))
iterate(E::TruncEigen, ::Val{:l}) = (E.l, Val(:r))
iterate(E::TruncEigen, ::Val{:r}) = (E.r, Val(:done))
iterate(E::TruncEigen, ::Val{:done}) = nothing

"""
    eigen(A::ITensor[, Linds, Rinds]; <keyword arguments>)

Eigendecomposition of an ITensor `A`, computed
by treating the "left indices" `Linds` provided collectively
as a row index, and remaining "right indices" `Rinds` as a
column index (matricization of a tensor).

If no indices are provided, pairs of primed and unprimed indices are
searched for, with `Linds` taken to be the primed indices and
`Rinds` taken to be the unprimed indices.

The return arguments are the eigenvalues `D` and eigenvectors `U`
as tensors, such that `A * U ∼ U * D` (more precisely they are approximately
equal up to proper replacements of indices, see the example for details).

Whether or not `eigen` performs a trunction depends on the keyword
arguments provided. Note that truncation is only well defined for
positive semidefinite matrices.

# Arguments

    - `maxdim::Int`: the maximum number of singular values to keep.
    - `mindim::Int`: the minimum number of singular values to keep.
    - `cutoff::Float64`: set the desired truncation error of the eigenvalues,
       by default defined as the sum of the squares of the smallest eigenvalues.
       For now truncation is only well defined for positive semi-definite
       eigenspectra.
    - `ishermitian::Bool = false`: specify if the matrix is Hermitian, in which
       case a specialized diagonalization routine will be used and it is
       guaranteed that real eigenvalues will be returned.
    - `plev::Int = 0`: set the prime level of the Indices of `D`. Default prime
       levels are subject to change.
    - `leftplev::Int = plev`: set the prime level of the Index unique to `D`.
       Default prime levels are subject to change.
    - `rightplev::Int = leftplev+1`: set the prime level of the Index shared
       by `D` and `U`. Default tags are subject to change.
    - `tags::String = "Link,eigen"`: set the tags of the Indices of `D`.
       Default tags are subject to change.
    - `lefttags::String = tags`: set the tags of the Index unique to `D`.
       Default tags are subject to change.
    - `righttags::String = tags`: set the tags of the Index shared by `D` and `U`.
       Default tags are subject to change.
    - `use_absolute_cutoff::Bool = false`: set if all probability weights below
       the `cutoff` value should be discarded, rather than the sum of discarded
       weights.
    - `use_relative_cutoff::Bool = true`: set if the singular values should
       be normalized for the sake of truncation.

# Examples

```julia
i, j, k, l = Index(2, "i"), Index(2, "j"), Index(2, "k"), Index(2, "l")
A = random_itensor(i, j, k, l)
Linds = (i, k)
Rinds = (j, l)
D, U = eigen(A, Linds, Rinds)
dl, dr = uniqueind(D, U), commonind(D, U)
Ul = replaceinds(U, (Rinds..., dr) => (Linds..., dl))
A * U ≈ Ul * D # true
```

See also: [`svd`](@ref), [`factorize`](@ref)
"""
function eigen(
        A::ITensor,
        Linds,
        Rinds;
        mindim = nothing,
        maxdim = nothing,
        cutoff = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = nothing,
        ishermitian = nothing,
        tags = nothing,
        lefttags = nothing,
        righttags = nothing,
        plev = nothing,
        leftplev = nothing,
        rightplev = nothing,
    )
    ishermitian = NDTensors.replace_nothing(ishermitian, false)
    tags = NDTensors.replace_nothing(tags, ts"Link,eigen")
    lefttags = NDTensors.replace_nothing(lefttags, tags)
    righttags = NDTensors.replace_nothing(righttags, tags)
    plev = NDTensors.replace_nothing(plev, 0)
    leftplev = NDTensors.replace_nothing(leftplev, plev)
    rightplev = NDTensors.replace_nothing(rightplev, plev)

    @debug_check begin
        if hasqns(A)
            @assert flux(A) == QN()
        end
    end

    N = ndims(A)
    NL = length(Linds)
    NR = length(Rinds)
    NL != NR && error("Must have equal number of left and right indices")
    N != NL + NR &&
        error("Number of left and right indices must add up to total number of indices")

    if lefttags == righttags && leftplev == rightplev
        leftplev = rightplev + 1
    end

    # Linds, Rinds may not have the correct directions
    Lis = indices(Linds)
    Ris = indices(Rinds)

    # Ensure the indices have the correct directions,
    # QNs, etc.
    # First grab the indices in A, then permute them
    # correctly.
    Lis = permute(commoninds(A, Lis), Lis)
    Ris = permute(commoninds(A, Ris), Ris)

    for (l, r) in zip(Lis, Ris)
        if space(l) != space(r)
            error("In eigen, indices must come in pairs with equal spaces.")
        end
        if hasqns(A)
            if dir(l) == dir(r)
                error("In eigen, indices must come in pairs with opposite directions")
            end
        end
    end

    # <fermions>
    L_arrow_dir = Out
    if hasqns(A) && using_auto_fermion()
        # Make arrows of combined ITensor match those of index sets
        if all(i -> dir(i) == Out, Lis) && all(i -> dir(i) == In, Ris)
            L_arrow_dir = Out
        elseif all(i -> dir(i) == In, Lis) && all(i -> dir(i) == Out, Ris)
            L_arrow_dir = In
        else
            error(
                "With auto_fermion enabled, index sets in eigen must have all arrows the same, and opposite between the sets",
            )
        end
    end

    CL = combiner(Lis...; dir = L_arrow_dir, tags = "CMB,left")
    CR = combiner(dag(Ris)...; dir = L_arrow_dir, tags = "CMB,right")

    AC = A * dag(CR) * CL

    cL = combinedind(CL)
    cR = dag(combinedind(CR))
    if inds(AC) != (cL, cR)
        AC = permute(AC, cL, cR)
    end

    AT = ishermitian ? Hermitian(tensor(AC)) : tensor(AC)

    DT, VT, spec = eigen(AT; mindim, maxdim, cutoff, use_absolute_cutoff, use_relative_cutoff)
    D, VC = itensor(DT), itensor(VT)

    V = VC * dag(CR)

    # Set right index tags
    l = uniqueind(D, V)
    r = commonind(D, V)
    l̃ = setprime(settags(l, lefttags), leftplev)
    r̃ = setprime(settags(l̃, righttags), rightplev)

    replaceinds!(D, (l, r), (l̃, r̃))
    replaceind!(V, r, r̃)

    l, r = l̃, r̃

    # The right eigenvectors, after being applied to A
    Vt = replaceinds(V, (Ris..., r), (Lis..., l))

    @debug_check begin
        if hasqns(A)
            @assert flux(D) == QN()
            @assert flux(V) == QN()
            @assert flux(Vt) == QN()
        end
    end

    return TruncEigen(D, V, Vt, spec, l, r)
end

function eigen(A::ITensor; kwargs...)
    Ris = filterinds(A; plev = 0)
    Lis = Ris'
    return eigen(A, Lis, Ris; kwargs...)
end

# ----------------------------- QR/RQ/QL/LQ decompositions ------------------------------

#
#  Helper functions for handleing cases where zero indices are requested on Q or R.
#
function add_trivial_index(A::ITensor, Ainds)
    α = trivial_index(Ainds) #If Ainds[1] has no QNs makes Index(1), otherwise Index(QN()=>1)
    vα = onehot(datatype(A), α => 1)
    A *= vα
    return A, vα, [α]
end

function add_trivial_index(A::ITensor, Linds, Rinds)
    vαl, vαr = nothing, nothing
    if isempty(Linds)
        A, vαl, Linds = add_trivial_index(A, Rinds)
    end
    if isempty(Rinds)
        A, vαr, Rinds = add_trivial_index(A, Linds)
    end
    return A, vαl, vαr, Linds, Rinds
end

remove_trivial_index(Q::ITensor, R::ITensor, vαl, vαr) = (Q * dag(vαl), R * dag(vαr))
remove_trivial_index(Q::ITensor, R::ITensor, ::Nothing, vαr) = (Q, R * dag(vαr))
remove_trivial_index(Q::ITensor, R::ITensor, vαl, ::Nothing) = (Q * dag(vαl), R)
remove_trivial_index(Q::ITensor, R::ITensor, ::Nothing, ::Nothing) = (Q, R)

#
#  Force users to knowingly ask for zero indices using qr(A,()) syntax
#
function noinds_error_message(decomp::String)
    return "$decomp without any input indices is currently not defined.
 In the future it may be defined as performing a $decomp decomposition
 treating the ITensor as a matrix from the primed to the unprimed indices."
end

qr(A::ITensor; kwargs...) = error(noinds_error_message("qr"))
rq(A::ITensor; kwargs...) = error(noinds_error_message("rq"))
lq(A::ITensor; kwargs...) = error(noinds_error_message("lq"))
ql(A::ITensor; kwargs...) = error(noinds_error_message("ql"))
#
# User supplied only left indices as a tuple or vector.
#
qr(A::ITensor, Linds::Indices; kwargs...) = qr(A, Linds, uniqueinds(A, Linds); kwargs...)
ql(A::ITensor, Linds::Indices; kwargs...) = ql(A, Linds, uniqueinds(A, Linds); kwargs...)
rq(A::ITensor, Linds::Indices; kwargs...) = rq(A, Linds, uniqueinds(A, Linds); kwargs...)
lq(A::ITensor, Linds::Indices; kwargs...) = lq(A, Linds, uniqueinds(A, Linds); kwargs...)
#
# User supplied only left indices as as vararg
#
qr(A::ITensor, Linds...; kwargs...) = qr(A, Linds, uniqueinds(A, Linds); kwargs...)
ql(A::ITensor, Linds...; kwargs...) = ql(A, Linds, uniqueinds(A, Linds); kwargs...)
rq(A::ITensor, Linds...; kwargs...) = rq(A, Linds, uniqueinds(A, Linds); kwargs...)
lq(A::ITensor, Linds...; kwargs...) = lq(A, Linds, uniqueinds(A, Linds); kwargs...)
#
# Core function where both left and right indices are supplied as tuples or vectors
# Handle default tags and dispatch to generic qx/xq functions.
#
function qr(A::ITensor, Linds::Indices, Rinds::Indices; tags = ts"Link,qr", kwargs...)
    return qx(qr, A, Linds, Rinds; tags, kwargs...)
end
function ql(A::ITensor, Linds::Indices, Rinds::Indices; tags = ts"Link,ql", kwargs...)
    return qx(ql, A, Linds, Rinds; tags, kwargs...)
end
function rq(A::ITensor, Linds::Indices, Rinds::Indices; tags = ts"Link,rq", kwargs...)
    return xq(ql, A, Linds, Rinds; tags, kwargs...)
end
function lq(A::ITensor, Linds::Indices, Rinds::Indices; tags = ts"Link,lq", kwargs...)
    return xq(qr, A, Linds, Rinds; tags, kwargs...)
end
#
#  Generic function implementing both qr and ql decomposition. The X tensor = R or L.
#
function qx(
        qx::Function, A::ITensor, Linds::Indices, Rinds::Indices; tags = ts"Link,qx", positive = false
    )
    # Strip out any extra indices that are not in A.
    # Unit test test/base/test_itensor.jl line 1469 will fail without this.
    Linds = commoninds(A, Linds)
    #Rinds=commoninds(A,Rinds) #if the user supplied Rinds they could have the same problem?
    #
    # Make a dummy index with dim=1 and incorporate into A so the Linds & Rinds can never
    # be empty.  A essentially becomes 1D after collection.
    #
    A, vαl, vαr, Linds, Rinds = add_trivial_index(A, Linds, Rinds)

    #
    #  Use combiners to render A down to a rank 2 tensor ready for matrix QR/QL routine.
    #
    CL, CR = combiner(Linds...), combiner(Rinds...)
    cL, cR = combinedind(CL), combinedind(CR)
    AC = A * CR * CL

    #
    #  Make sure we don't accidentally pass the transpose into the matrix qr/ql routine.
    #
    AC = permute(AC, cL, cR; allow_alias = true)

    QT, XT = qx(tensor(AC); positive) #pass order(AC)==2 matrix down to the NDTensors level where qr/ql are implemented.
    #
    #  Undo the combine oepration, to recover all tensor indices.
    #
    Q, X = itensor(QT) * dag(CL), itensor(XT) * dag(CR)

    # Remove dummy indices.  No-op if vαl and vαr are Nothing
    Q, X = remove_trivial_index(Q, X, vαl, vαr)
    #
    # fix up the tag name for the index between Q and X.
    #
    q = commonind(Q, X)
    Q = settags(Q, tags, q)
    X = settags(X, tags, q)
    q = settags(q, tags)

    return Q, X, q
end

#
#  Generic function implementing both rq and lq decomposition. Implemented using qr/ql
#  with swapping the left and right indices.  The X tensor = R or L.
#
function xq(
        qx::Function, A::ITensor, Linds::Indices, Rinds::Indices; tags = ts"Link,xq", positive = false
    )
    Q, X, q = qx(A, Rinds, Linds; positive)
    #
    # fix up the tag name for the index between Q and L.
    #
    Q = settags(Q, tags, q)
    X = settags(X, tags, q)
    q = settags(q, tags)

    return X, Q, q
end

polar(A::ITensor; kwargs...) = error(noinds_error_message("polar"))

# TODO: allow custom tags in internal indices?
# TODO: return the new common indices?
function polar(A::ITensor, Linds...)
    U, S, V = svd(A, Linds...)
    u = commoninds(S, U)
    v = commoninds(S, V)
    δᵤᵥ′ = δ(eltype(A), u..., v'...)
    Q = U * δᵤᵥ′ * V'
    P = dag(V') * dag(δᵤᵥ′) * S * V
    return Q, P, commoninds(Q, P)
end

function factorize_qr(A::ITensor, Linds...; ortho = "left", tags = nothing, positive = false)
    if ortho == "left"
        L, R, q = qr(A, Linds...; tags, positive)
    elseif ortho == "right"
        Lis = uniqueinds(A, indices(Linds...))
        R, L, q = qr(A, Lis...; tags, positive)
    else
        error("In factorize using qr decomposition, ortho keyword
$ortho not supported. Supported options are left or right.")
    end
    return L, R
end

using NDTensors: map_diag!
#
# Generic function implementing a square root decomposition of a diagonal, order 2 tensor with inds u, v
#
function sqrt_decomp(D::ITensor, u::Index, v::Index)
    (storage(D) isa Union{Diag, DiagBlockSparse}) ||
        error("Must be a diagonal matrix ITensor.")
    sqrtDL = adapt(datatype(D), diag_itensor(u, dag(u)'))
    sqrtDR = adapt(datatype(D), diag_itensor(v, dag(v)'))
    map_diag!(sqrt ∘ abs, sqrtDL, D)
    map_diag!(sqrt ∘ abs, sqrtDR, D)
    δᵤᵥ = copy(D)
    map_diag!(sign, δᵤᵥ, D)
    return sqrtDL, prime(δᵤᵥ), sqrtDR
end

function factorize_svd(
        A::ITensor,
        Linds...;
        (singular_values!) = nothing,
        ortho = "left",
        alg = nothing,
        dir = nothing,
        mindim = nothing,
        maxdim = nothing,
        cutoff = nothing,
        tags = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = nothing,
        min_blockdim = nothing,
    )
    leftdir, rightdir = dir, dir
    if !isnothing(leftdir)
        leftdir = -leftdir
    end
    if !isnothing(rightdir)
        rightdir = -rightdir
    end
    USV = svd(
        A,
        Linds...;
        leftdir,
        rightdir,
        alg,
        mindim,
        maxdim,
        cutoff,
        lefttags = tags,
        righttags = tags,
        use_absolute_cutoff,
        use_relative_cutoff,
        min_blockdim,
    )
    if isnothing(USV)
        return nothing
    end
    U, S, V, spec, u, v = USV
    if ortho == "left"
        L, R = U, S * V
    elseif ortho == "right"
        L, R = U * S, V
    elseif ortho == "none"
        sqrtDL, δᵤᵥ, sqrtDR = sqrt_decomp(S, u, v)
        sqrtDR = denseblocks(sqrtDR) * denseblocks(δᵤᵥ)
        L, R = U * sqrtDL, V * sqrtDR
    else
        error("In factorize using svd decomposition, ortho keyword
$ortho not supported. Supported options are left, right, or none.")
    end

    !isnothing(singular_values!) && (singular_values![] = S)

    return L, R, spec
end

function factorize_eigen(
        A::ITensor,
        Linds...;
        ortho = "left",
        eigen_perturbation = nothing,
        mindim = nothing,
        maxdim = nothing,
        cutoff = nothing,
        tags = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = nothing,
    )
    if ortho == "left"
        Lis = commoninds(A, indices(Linds...))
    elseif ortho == "right"
        Lis = uniqueinds(A, indices(Linds...))
    else
        error("In factorize using eigen decomposition, ortho keyword
$ortho not supported. Supported options are left or right.")
    end
    simLis = sim(Lis)
    A2 = A * replaceinds(dag(A), Lis, simLis)
    if !isnothing(eigen_perturbation)
        # This assumes delta_A2 has indices:
        # (Lis..., prime(Lis)...)
        delta_A2 = replaceinds(eigen_perturbation, Lis, dag(simLis))
        delta_A2 = noprime(delta_A2)
        A2 += delta_A2
    end
    F = eigen(
        A2,
        Lis,
        simLis;
        ishermitian = true,
        mindim,
        maxdim,
        cutoff,
        tags,
        use_absolute_cutoff,
        use_relative_cutoff,
    )
    D, _, spec = F
    L = F.Vt
    R = dag(L) * A
    if ortho == "right"
        L, R = R, L
    end
    return L, R, spec
end

factorize(A::ITensor; kwargs...) = error(noinds_error_message("factorize"))

"""
    factorize(A::ITensor, Linds::Index...; <keyword arguments>)

Perform a factorization of `A` into ITensors `L` and `R` such that `A ≈ L * R`.

# Arguments

- `ortho::String = "left"`: Choose orthogonality
   properties of the factorization.
    + `"left"`: the left factor `L` is an orthogonal basis
       such that `L * dag(prime(L, commonind(L,R))) ≈ I`.
    + `"right"`: the right factor `R` forms an orthogonal basis.
    + `"none"`, neither of the factors form an orthogonal basis,
        and in general are made as symmetrically as possible
        (depending on the decomposition used).
- `which_decomp::Union{String, Nothing} = nothing`: choose what kind
   of decomposition is used.
    + `nothing`: choose the decomposition automatically based on
       the other arguments. For example, when `nothing` is chosen and
       `ortho = "left"` or `"right"`, and a cutoff is provided, `svd` or
       `eigen` is used depending on the provided cutoff (`eigen` is only
       used when the cutoff is greater than `1e-12`, since it has a lower
       precision). When no truncation is requested `qr` is used for dense
       ITensors and `svd` for block-sparse ITensors (in the future `qr`
       will be used also for block-sparse ITensors in this case).
    + `"svd"`: `L = U` and `R = S * V` for `ortho = "left"`, `L = U * S`
       and `R = V` for `ortho = "right"`, and `L = U * sqrt.(S)` and
       `R = sqrt.(S) * V` for `ortho = "none"`. To control which `svd`
       algorithm is choose, use the `svd_alg` keyword argument. See the
       documentation for `svd` for the supported algorithms, which are the
       same as those accepted by the `alg` keyword argument.
    + `"eigen"`: `L = U` and ``R = U^{\\dagger} A`` where `U` is determined
       from the eigendecompositon ``A A^{\\dagger} = U D U^{\\dagger}`` for
       `ortho = "left"` (and vice versa for `ortho = "right"`). `"eigen"` is
       not supported for `ortho = "none"`.
    + `"qr"`: `L=Q` and `R` an upper-triangular matrix when
       `ortho = "left"`, and `R = Q` and `L` a lower-triangular matrix
       when `ortho = "right"` (currently supported for dense ITensors only).
      In the future, other decompositions like QR (for block-sparse ITensors),
      polar, cholesky, LU, etc. are expected to be supported.

For truncation arguments, see: [`svd`](@ref)
"""
function factorize(
        A::ITensor,
        Linds...;
        mindim = nothing,
        maxdim = nothing,
        cutoff = nothing,
        ortho = nothing,
        tags = nothing,
        plev = nothing,
        which_decomp = nothing,
        # eigen
        eigen_perturbation = nothing,
        # svd
        svd_alg = nothing,
        use_absolute_cutoff = nothing,
        use_relative_cutoff = nothing,
        min_blockdim = nothing,
        (singular_values!) = nothing,
        dir = nothing,
    )
    @debug_check checkflux(A)
    if !isnothing(eigen_perturbation)
        if !(isnothing(which_decomp) || which_decomp == "eigen")
            error(
                """when passing a non-trivial eigen_perturbation to `factorize`,
                the which_decomp keyword argument must be either "automatic" or
                "eigen" """
            )
        end
        which_decomp = "eigen"
    end
    ortho = NDTensors.replace_nothing(ortho, "left")
    tags = NDTensors.replace_nothing(tags, ts"Link,fact")
    plev = NDTensors.replace_nothing(plev, 0)

    # Determines when to use eigen vs. svd (eigen is less precise,
    # so eigen should only be used if a larger cutoff is requested)
    automatic_cutoff = 1.0e-12
    Lis = commoninds(A, indices(Linds...))
    Ris = uniqueinds(A, Lis)
    dL, dR = dim(Lis), dim(Ris)
    if isnothing(eigen_perturbation)
        # maxdim is forced to be at most the max given SVD
        if isnothing(maxdim)
            maxdim = min(dL, dR)
        end
        maxdim = min(maxdim, min(dL, dR))
    else
        if isnothing(maxdim)
            maxdim = max(dL, dR)
        end
        maxdim = min(maxdim, max(dL, dR))
    end
    might_truncate = !isnothing(cutoff) || maxdim < min(dL, dR)
    if isnothing(which_decomp)
        if !might_truncate && ortho != "none"
            which_decomp = "qr"
        elseif isnothing(cutoff) || cutoff ≤ automatic_cutoff
            which_decomp = "svd"
        elseif cutoff > automatic_cutoff
            which_decomp = "eigen"
        end
    end
    if which_decomp == "svd"
        LR = factorize_svd(
            A,
            Linds...;
            mindim,
            maxdim,
            cutoff,
            tags,
            ortho,
            alg = svd_alg,
            dir,
            singular_values!,
            use_absolute_cutoff,
            use_relative_cutoff,
            min_blockdim,
        )
        if isnothing(LR)
            return nothing
        end
        L, R, spec = LR
    elseif which_decomp == "eigen"
        L, R, spec = factorize_eigen(
            A,
            Linds...;
            mindim,
            maxdim,
            cutoff,
            tags,
            ortho,
            eigen_perturbation,
            use_absolute_cutoff,
            use_relative_cutoff,
        )
    elseif which_decomp == "qr"
        L, R = factorize_qr(A, Linds...; ortho, tags)
        spec = Spectrum(nothing, 0.0)
    else
        throw(
            ArgumentError(
                """In factorize, factorization $which_decomp is not
                currently supported. Use `"svd"`, `"eigen"`, `"qr"` or `nothing`."""
            )
        )
    end

    # Set the tags and prime level
    l = commonind(L, R)
    l̃ = setprime(settags(l, tags), plev)
    L = replaceind(L, l, l̃)
    R = replaceind(R, l, l̃)
    l = l̃

    return L, R, spec, l
end
