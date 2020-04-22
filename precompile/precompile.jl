const __bodyfunction__ = Dict{Method,Any}()

# Find keyword "body functions" (the function that contains the body
# as written by the developer, called after all missing keyword-arguments
# have been assigned values), in a manner that doesn't depend on
# gensymmed names.
# `mnokw` is the method that gets called when you invoke it without
# supplying any keywords.
function __lookup_kwbody__(mnokw::Method)
    function getsym(arg)
        isa(arg, Symbol) && return arg
        @assert isa(arg, GlobalRef)
        return arg.name
    end

    f = get(__bodyfunction__, mnokw, nothing)
    if f === nothing
        fmod = mnokw.module
        # The lowered code for `mnokw` should look like
        #   %1 = mkw(kwvalues..., #self#, args...)
        #        return %1
        # where `mkw` is the name of the "active" keyword body-function.
        ast = Base.uncompressed_ast(mnokw)
        if isa(ast, Core.CodeInfo) && length(ast.code) >= 2
            callexpr = ast.code[end-1]
            if isa(callexpr, Expr) && callexpr.head == :call
                fsym = callexpr.args[1]
                if isa(fsym, Symbol)
                    f = getfield(fmod, fsym)
                elseif isa(fsym, GlobalRef)
                    if fsym.mod === Core && fsym.name === :_apply
                        f = getfield(mnokw.module, getsym(callexpr.args[2]))
                    elseif fsym.mod === Core && fsym.name === :_apply_iterate
                        f = getfield(mnokw.module, getsym(callexpr.args[3]))
                    else
                        f = getfield(fsym.mod, fsym.name)
                    end
                else
                    f = missing
                end
            else
                f = missing
            end
        else
            f = missing
        end
        __bodyfunction__[mnokw] = f
    end
    return f
end

function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    let fbody = try __lookup_kwbody__(which(ITensors.addtags, (ITensor{3},String,Vararg{Any,N} where N,))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},typeof(addtags),ITensor{3},String,Vararg{Any,N} where N,))
        end
    end
    let fbody = try __lookup_kwbody__(which(ITensors.op, (Index{Int64},String,))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},typeof(op),Index{Int64},String,))
        end
    end
    let fbody = try __lookup_kwbody__(which(ITensors.prime, (ITensor{3},Int64,Vararg{Any,N} where N,))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Base.Iterators.Pairs{Union{},Union{},Tuple{},NamedTuple{(),Tuple{}}},typeof(prime),ITensor{3},Int64,Vararg{Any,N} where N,))
        end
    end
    let fbody = try __lookup_kwbody__(which(sortperm!, (Array{Int64,1},Array{ITensors.SiteOp,1},))) catch missing end
        if !ismissing(fbody)
            precompile(fbody, (Base.Sort.InsertionSortAlg,Function,Function,Nothing,Base.Order.ForwardOrdering,Bool,typeof(sortperm!),Array{Int64,1},Array{ITensors.SiteOp,1},))
        end
    end
    precompile(Tuple{Core.kwftype(typeof(ITensors.NDTensors.eigen)),NamedTuple{(:ishermitian, :which_decomp, :tags, :maxdim, :mindim, :cutoff, :eigen_perturbation, :ortho),Tuple{Bool,String,TagSet,Int64,Int64,Float64,Nothing,String}},typeof(eigen),LinearAlgebra.Hermitian{Float64,ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}}}})
    precompile(Tuple{Core.kwftype(typeof(ITensors.factorize)),NamedTuple{(:which_decomp, :tags, :maxdim, :mindim, :cutoff, :eigen_perturbation, :ortho),Tuple{String,TagSet,Int64,Int64,Float64,Nothing,String}},typeof(factorize),ITensor{3},IndexSet{2,Index{Int64}}})
    precompile(Tuple{Core.kwftype(typeof(ITensors.replacebond!)),NamedTuple{(:maxdim, :mindim, :cutoff, :eigen_perturbation, :ortho, :which_decomp),Tuple{Int64,Int64,Float64,Nothing,String,String}},typeof(replacebond!),MPS,Int64,ITensor{3}})
    precompile(Tuple{Core.kwftype(typeof(KrylovKit.eigsolve)),NamedTuple{(:ishermitian, :tol, :krylovdim, :maxiter),Tuple{Bool,Float64,Int64,Int64}},typeof(KrylovKit.eigsolve),ProjMPO,ITensor{3},Int64,Symbol})
    precompile(Tuple{Core.kwftype(typeof(KrylovKit.expand!)),NamedTuple{(:verbosity,),Tuple{Int64}},typeof(KrylovKit.expand!),KrylovKit.LanczosIterator{ProjMPO,ITensor{3},KrylovKit.ModifiedGramSchmidt2},KrylovKit.LanczosFactorization{ITensor{3},Float64}})
    precompile(Tuple{ProjMPO,ITensor{3}})
    precompile(Tuple{Type{MPO},AutoMPO,Array{Index{Int64},1}})
    precompile(Tuple{Type{TagSet},String})
    precompile(Tuple{typeof(*),Float64,ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}}})
    precompile(Tuple{typeof(*),ITensor{2},ITensor{3}})
    precompile(Tuple{typeof(*),ITensor{3},ITensor{1}})
    precompile(Tuple{typeof(*),ITensor{3},ITensor{2}})
    precompile(Tuple{typeof(*),ITensor{3},ITensor{3}})
    precompile(Tuple{typeof(*),ITensor{4},ITensor{1}})
    precompile(Tuple{typeof(*),ITensor{4},ITensor{4}})
    precompile(Tuple{typeof(==),Array{Index{Int64},1},Array{Index{Int64},1}})
    precompile(Tuple{typeof(==),IndexSet{1,Index{Int64}},IndexSet{1,Index{Int64}}})
    precompile(Tuple{typeof(==),IndexSet{2,Index{Int64}},IndexSet{2,Index{Int64}}})
    precompile(Tuple{typeof(==),IndexSet{3,Index{Int64}},IndexSet{3,Index{Int64}}})
    precompile(Tuple{typeof(==),IndexVal{Index{Int64}},Index{Int64}})
    precompile(Tuple{typeof(==),TagSet,TagSet})
    precompile(Tuple{typeof(Base.Broadcast.materialize!),ITensor{4},Base.Broadcast.Broadcasted{ITensors.ITensorStyle,Nothing,typeof(+),Tuple{ITensor{4},ITensor{4}}}})
    precompile(Tuple{typeof(Base.afoldl),typeof(*),ITensor{4},ITensor{3}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Diag{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,0,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{0,Union{}}},Tuple{},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Diag{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,1,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{1,Index{Int64}}},Tuple{Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,1,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{1,Index{Int64}}},Tuple{Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contract!!),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64}})
    precompile(Tuple{typeof(ITensors.NDTensors.contraction_output),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},IndexSet{2,Index{Int64}}})
    precompile(Tuple{typeof(ITensors.NDTensors.getperm),IndexSet{4,Index{Int64}},IndexSet{4,Index{Int64}}})
    precompile(Tuple{typeof(ITensors.NDTensors.permutedims!!),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},Function})
    precompile(Tuple{typeof(ITensors.NDTensors.permutedims!!),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},Function})
    precompile(Tuple{typeof(ITensors.compute_contraction_labels),IndexSet{2,Index{Int64}},IndexSet{3,Index{Int64}}})
    precompile(Tuple{typeof(ITensors.compute_contraction_labels),IndexSet{4,Index{Int64}},IndexSet{3,Index{Int64}}})
    precompile(Tuple{typeof(KrylovKit.orthogonalize!),ITensor{3},ITensor{3},KrylovKit.ModifiedGramSchmidt2})
    precompile(Tuple{typeof(Random.randn!),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}}})
    precompile(Tuple{typeof(axpy!),Float64,ITensor{3},ITensor{3}})
    precompile(Tuple{typeof(combiner),Index{Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Number,2,ITensors.NDTensors.Combiner,IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Number,2,ITensors.NDTensors.Combiner,IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},NTuple{4,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Number,2,ITensors.NDTensors.Combiner,IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Number,2,ITensors.NDTensors.Combiner,IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Diag{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Diag{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Diag{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Diag{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,1,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{1,Index{Int64}}},Tuple{Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,1,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{1,Index{Int64}}},Tuple{Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},NTuple{4,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},Tuple{Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},Tuple{}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,1,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{1,Index{Int64}}},Tuple{Int64},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,1,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{1,Index{Int64}}},Tuple{Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,3,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{3,Index{Int64}}},Tuple{Int64,Int64,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},NTuple{4,Int64}})
    precompile(Tuple{typeof(contract),ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64},ITensors.NDTensors.Tensor{Float64,4,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{4,Index{Int64}}},NTuple{4,Int64}})
    precompile(Tuple{typeof(dmrg),MPO,MPS,Sweeps})
    precompile(Tuple{typeof(hassameinds),ITensor{3},Tuple{Index{Int64},Index{Int64},Index{Int64}}})
    precompile(Tuple{typeof(hastags),TagSet,String})
    precompile(Tuple{typeof(itensor),Array{Float64,3},Index{Int64},Index{Int64},Vararg{Index{Int64},N} where N})
    precompile(Tuple{typeof(mul!),ITensor{3},ITensor{3},Float64})
    precompile(Tuple{typeof(prime),ITensor{3},ITensors.Not{TagSet}})
    precompile(Tuple{typeof(randomITensor),Index{Int64},Index{Int64},Index{Int64}})
    precompile(Tuple{typeof(randomMPS),Array{Index{Int64},1},Int64})
    precompile(Tuple{typeof(replaceind!),ITensor{2},Index{Int64},Index{Int64}})
    precompile(Tuple{typeof(replaceind),IndexSet{2,Index{Int64}},Index{Int64},Index{Int64}})
    precompile(Tuple{typeof(setdiff),IndexSet{2,Index{Int64}},IndexSet{1,Index{Int64}}})
    precompile(Tuple{typeof(setindex!),Dict{Array{ITensors.SiteOp,1},Array{Float64,2}},Array{Float64,2},Array{ITensors.SiteOp,1}})
    precompile(Tuple{typeof(show),Base.GenericIOBuffer{Array{UInt8,1}},TagSet})
    precompile(Tuple{typeof(svd),ITensors.NDTensors.Tensor{Complex{Float64},2,ITensors.NDTensors.Dense{Complex{Float64},Array{Complex{Float64},1}},IndexSet{2,Index{Int64}}}})
    precompile(Tuple{typeof(svd),ITensors.NDTensors.Tensor{Float64,2,ITensors.NDTensors.Dense{Float64,Array{Float64,1}},IndexSet{2,Index{Int64}}}})
end
