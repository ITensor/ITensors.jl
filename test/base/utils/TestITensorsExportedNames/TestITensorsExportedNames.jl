module TestITensorsExportedNames
#=
# List constructed with (along with cleaning up
# macro symbols):
using DelimitedFiles: writedlm
using ITensors: ITensors
open("itensors_exported_names.jl", "w") do io
  writedlm(io, repr.(names(ITensors)) .* ",")
end
=#
const ITENSORMPS_EXPORTED_NAMES = [
  Symbol("@OpName_str"),
  Symbol("@SiteType_str"),
  Symbol("@StateName_str"),
  Symbol("@TagType_str"),
  Symbol("@ValName_str"),
  Symbol("@preserve_ortho"),
  Symbol("@visualize"),
  Symbol("@visualize!"),
  Symbol("@visualize_noeval"),
  Symbol("@visualize_noeval!"),
  Symbol("@visualize_sequence"),
  Symbol("@visualize_sequence_noeval"),
  :AbstractMPS,
  :AbstractObserver,
  :Apply,
  :AutoMPO,
  :DMRGMeasurement,
  :DMRGObserver,
  :ITensorMPS,
  :Lattice,
  :LatticeBond,
  :MPO,
  :MPS,
  :NoObserver,
  :Op,
  :OpName,
  :OpSum,
  :Ops,
  :Prod,
  :ProjMPO,
  :ProjMPOSum,
  :ProjMPO_MPS,
  :Scaled,
  :SiteType,
  :Spectrum,
  :StateName,
  :Sum,
  :Sweeps,
  :Trotter,
  :ValName,
  :add,
  :add!,
  :apply,
  :applyMPO,
  :applympo,
  :argsdict,
  :checkdone!, # remove export
  :coefficient,
  :common_siteind,
  :common_siteinds,
  :correlation_matrix,
  :cutoff,
  :cutoff!, # deprecate
  :disk,
  :dmrg,
  :dot, # remove export
  :eigs, # deprecate
  :energies, # deprecate
  :entropy, # deprecate
  :errorMPOprod, # deprecate
  :error_contract,
  :error_mpoprod, # deprecate
  :error_mul, # deprecate
  :expect,
  :findfirstsiteind, # deprecate
  :findfirstsiteinds, # deprecate
  :findsite, # deprecate
  :findsites, # deprecate
  :firstsiteind, # deprecate
  :firstsiteinds, # deprecate
  :get_cutoffs, # deprecate
  :get_maxdims, # deprecate
  :get_mindims, # deprecate
  :get_noises, # deprecate
  :has_fermion_string, # remove export
  :hassameinds,
  :inner,
  :isortho,
  :linkdim,
  :linkdims,
  :linkind,
  :linkindex,
  :linkinds,
  :logdot,
  :loginner,
  :lognorm,
  :lproj,
  :maxdim,
  :maxdim!,
  :maxlinkdim,
  :measure!,
  :measurements,
  :mindim,
  :mindim!,
  :movesite,
  :movesites,
  :mul, # deprecate
  :multMPO,
  :multmpo,
  :noise,
  :noise!,
  :noiseterm,
  :nsite,
  :nsweep,
  :op,
  :ops,
  :orthoCenter,
  :ortho_lims,
  :orthocenter,
  :orthogonalize,
  :orthogonalize!,
  :outer,
  :position!,
  :product,
  :primelinks!,
  :productMPS,
  :projector,
  :promote_itensor_eltype,
  :randomITensor,
  :randomMPO,
  :randomMPS,
  :replace_siteinds,
  :replace_siteinds!,
  :replacebond,
  :replacebond!,
  :replaceprime,
  :replacesites!,
  :reset_ortho_lims!,
  :rproj,
  :sample,
  :sample!,
  :set_leftlim!,
  :set_ortho_lims!,
  :set_rightlim!,
  :setcutoff!,
  :setmaxdim!,
  :setmindim!,
  :setnoise!,
  :simlinks!,
  :siteind,
  :siteindex,
  :siteinds,
  :splitblocks,
  :square_lattice,
  :state,
  :sum,
  :swapbondsites,
  :sweepnext,
  :tensors,
  :toMPO,
  :totalqn,
  :tr,
  :triangular_lattice,
  :truncate,
  :truncate!,
  :truncerror,
  :truncerrors,
  :unique_siteind,
  :unique_siteinds,
  :val,
  :⋅,
]
const ITENSOR_EXPORTED_NAMES = [
  Symbol("@disable_warn_order"),
  Symbol("@reset_warn_order"),
  Symbol("@set_warn_order"),
  Symbol("@ts_str"),
  :Block, # remove export
  :ITensor,
  :ITensors,
  :Index,
  :IndexSet, # deprecate
  :IndexVal, # deprecate
  :LinearAlgebra, # remove export
  :NDTensors, # remove export
  :Order, # deprecate
  :QN, # remove export
  :TagSet, # remove export
  :TagType, # deprecate
  :addblock!, # deprecate
  :addtags,
  :addtags!, # deprecate
  :allhastags, # deprecate
  :anyhastags, # deprecate
  :array, # deprecate
  :axpy!, # remove export
  :blockdim, # deprecate
  :blockoffsets, # deprecate
  :checkflux,
  :combinedind,
  :combiner,
  :commonind,
  :commonindex, # deprecate
  :commoninds,
  :complex!, # deprecate
  :contract,
  :convert_eltype, # remove export
  :convert_leaf_eltype, # remove export
  :dag,
  :delta,
  :dense,
  :denseblocks,
  :diag,
  :diagITensor,
  :diagitensor, # deprecate
  :dim, # remove export
  :dims, # remove export
  :dir,
  :directsum,
  :disable_combine_contract!, # deprecate
  :disable_tblis!, # deprecate
  :disable_warn_order!, # remove export
  :eachindval, # deprecate
  :eachnzblock, # deprecate
  :eachval, # deprecate
  :eigen, # remove export
  :emptyITensor, # deprecate
  :enable_combine_contract!, # deprecate
  :enable_tblis!, # deprecate
  :factorize, # remove export
  :filterinds,
  :findindex, # deprecate
  :findinds, # deprecate
  :firstind,
  :firstintersect, # deprecate
  :firstsetdiff, # deprecate
  :flux,
  :fparity, # remove export
  :getfirst, # remove export
  :getindex, # remove export
  :hadamard_product, # remove export
  :hascommoninds,
  :hasid,
  :hasind,
  :hasinds,
  :hasplev,
  :hasqns,
  :hastags,
  :id,
  :ind,
  :index_id_rng, # remove export
  :inds,
  :insertblock!, # remove export
  :isactive, # remove export
  :isfermionic, # remove export
  :ishermitian, # remove export
  :isindequal, # remove export
  :itensor,
  :lq, # remove export
  :mapprime,
  :mapprime!, # remove export (not defined?)
  :matmul, # deprecate
  :matrix, # deprecate
  :modulus, # remove export
  :mul!, # remove export
  :nblocks,
  :nnz,
  :nnzblocks,
  :noncommonind,
  :noncommoninds,
  :noprime,
  :noprime!, # deprecate
  :norm, # remove export
  :normalize, # remove export
  :normalize!, # remove export
  :not, # remove export
  :nullspace,
  :nzblock,
  :nzblocks,
  :onehot,
  :order,
  :permute,
  :plev,
  :polar,
  :pop,
  :popfirst,
  :prime,
  :prime!,
  :push,
  :pushfirst,
  :ql,
  :qn,
  :qr,
  :randn!, # remove export
  :readcpp,
  :removeqn,
  :removeqns,
  :removetags,
  :removetags!, # deprecate
  :replaceind,
  :replaceind!,
  :replaceindex!,
  :replaceinds,
  :replaceinds!,
  :replacetags,
  :replacetags!,
  :reset_warn_order!,
  :rmul!,
  :rq,
  :scalar,
  :scale!,
  :set_warn_order!,
  :setdir,
  :setelt,
  :setindex,
  :setprime,
  :setprime!,
  :setspace,
  :settags,
  :settags!,
  :sim,
  :sim!,
  :space,
  :storage,
  :store,
  :svd,
  :swapind,
  :swapinds,
  :swapinds!,
  :swapprime,
  :swapprime!,
  :swaptags,
  :swaptags!,
  :tags,
  :transpose,
  :unionind,
  :unioninds,
  :uniqueind,
  :uniqueindex,
  :uniqueinds,
  :use_combine_contract,
  :use_debug_checks,
  :vector,
  :δ,
  :⊕,
  :⊙,
]
const ITENSORS_EXPORTED_NAMES = [
  ITENSOR_EXPORTED_NAMES
  ITENSORMPS_EXPORTED_NAMES
]
end
