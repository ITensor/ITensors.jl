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
const ITENSORS_EXPORTED_NAMES = [
  Symbol("@OpName_str"),
  Symbol("@SiteType_str"),
  Symbol("@StateName_str"),
  Symbol("@TagType_str"),
  Symbol("@ValName_str"),
  Symbol("@disable_warn_order"),
  Symbol("@preserve_ortho"),
  Symbol("@reset_warn_order"),
  Symbol("@set_warn_order"),
  Symbol("@ts_str"),
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
  :Block,
  :DMRGMeasurement,
  :DMRGObserver,
  :ITensor,
  :ITensors,
  :Index,
  :IndexSet,
  :IndexVal,
  :Lattice,
  :LatticeBond,
  :LinearAlgebra,
  :MPO,
  :MPS,
  :NDTensors,
  :NoObserver,
  :Op,
  :OpName,
  :OpSum,
  :Ops,
  :Order,
  :Prod,
  :ProjMPO,
  :ProjMPOSum,
  :ProjMPO_MPS,
  :QN,
  :Scaled,
  :SiteType,
  :Spectrum,
  :StateName,
  :Sum,
  :Sweeps,
  :TagSet,
  :TagType,
  :Trotter,
  :ValName,
  :add,
  :add!,
  :addblock!,
  :addtags,
  :addtags!,
  :allhastags,
  :anyhastags,
  :apply,
  :applyMPO,
  :applympo,
  :argsdict,
  :array,
  :axpy!,
  :blockdim,
  :blockoffsets,
  :checkdone!,
  :checkflux,
  :coefficient,
  :combinedind,
  :combiner,
  :common_siteind,
  :common_siteinds,
  :commonind,
  :commonindex,
  :commoninds,
  :complex!,
  :contract,
  :convert_eltype,
  :convert_leaf_eltype,
  :correlation_matrix,
  :cutoff,
  :cutoff!,
  :dag,
  :delta,
  :dense,
  :denseblocks,
  :diag,
  :diagITensor,
  :diag_itensor,
  :diagitensor,
  :dim,
  :dims,
  :dir,
  :directsum,
  :disable_tblis!,
  :disable_warn_order!,
  :disk,
  :dmrg,
  :dot,
  :eachindval,
  :eachnzblock,
  :eachval,
  :eigen,
  :eigs,
  :emptyITensor,
  :enable_tblis!,
  :energies,
  :entropy,
  :errorMPOprod,
  :error_contract,
  :error_mpoprod,
  :error_mul,
  :expect,
  :factorize,
  :filterinds,
  :findfirstsiteind,
  :findfirstsiteinds,
  :findindex,
  :findinds,
  :findsite,
  :findsites,
  :firstind,
  :firstintersect,
  :firstsetdiff,
  :firstsiteind,
  :firstsiteinds,
  :flux,
  :fparity,
  :get_cutoffs,
  :get_maxdims,
  :get_mindims,
  :get_noises,
  :getfirst,
  :getindex,
  :hadamard_product,
  :has_fermion_string,
  :hascommoninds,
  :hasid,
  :hasind,
  :hasinds,
  :hasplev,
  :hasqns,
  :hassameinds,
  :hastags,
  :id,
  :ind,
  :index_id_rng,
  :inds,
  :inner,
  :insertblock!,
  :isactive,
  :isfermionic,
  :ishermitian,
  :isindequal,
  :isortho,
  :itensor,
  :linkdim,
  :linkdims,
  :linkind,
  :linkindex,
  :linkinds,
  :logdot,
  :loginner,
  :lognorm,
  :lproj,
  :lq,
  :mapprime,
  :mapprime!,
  :matmul,
  :matrix,
  :maxdim,
  :maxdim!,
  :maxlinkdim,
  :measure!,
  :measurements,
  :mindim,
  :mindim!,
  :modulus,
  :movesite,
  :movesites,
  :mul,
  :mul!,
  :multMPO,
  :multmpo,
  :nblocks,
  :nnz,
  :nnzblocks,
  :noise,
  :noise!,
  :noiseterm,
  :noncommonind,
  :noncommoninds,
  :noprime,
  :noprime!,
  :norm,
  :normalize,
  :normalize!,
  :not,
  :nsite,
  :nsweep,
  :nullspace,
  :nzblock,
  :nzblocks,
  :onehot,
  :op,
  :ops,
  :order,
  :orthoCenter,
  :ortho_lims,
  :orthocenter,
  :orthogonalize,
  :orthogonalize!,
  :outer,
  :permute,
  :plev,
  :polar,
  :pop,
  :popfirst,
  :position!,
  :prime,
  :prime!,
  :primelinks!,
  :product,
  :productMPS,
  :projector,
  :promote_itensor_eltype,
  :push,
  :pushfirst,
  :ql,
  :qn,
  :qr,
  :randn!,
  :randomITensor,
  :random_itensor,
  :randomMPO,
  :randomMPS,
  :readcpp,
  :removeqn,
  :removeqns,
  :removetags,
  :removetags!,
  :replace_siteinds,
  :replace_siteinds!,
  :replacebond,
  :replacebond!,
  :replaceind,
  :replaceind!,
  :replaceindex!,
  :replaceinds,
  :replaceinds!,
  :replaceprime,
  :replacesites!,
  :replacetags,
  :replacetags!,
  :reset_ortho_lims!,
  :reset_warn_order!,
  :rmul!,
  :rproj,
  :rq,
  :sample,
  :sample!,
  :scalar,
  :scale!,
  :set_leftlim!,
  :set_ortho_lims!,
  :set_rightlim!,
  :set_warn_order!,
  :setcutoff!,
  :setdir,
  :setelt,
  :setindex,
  :setmaxdim!,
  :setmindim!,
  :setnoise!,
  :setprime,
  :setprime!,
  :setspace,
  :settags,
  :settags!,
  :sim,
  :sim!,
  :simlinks!,
  :siteind,
  :siteindex,
  :siteinds,
  :space,
  :splitblocks,
  :square_lattice,
  :state,
  :storage,
  :store,
  :sum,
  :svd,
  :swapbondsites,
  :swapind,
  :swapinds,
  :swapinds!,
  :swapprime,
  :swapprime!,
  :swaptags,
  :swaptags!,
  :sweepnext,
  :tags,
  :tensors,
  :toMPO,
  :totalqn,
  :tr,
  :transpose,
  :triangular_lattice,
  :truncate,
  :truncate!,
  :truncerror,
  :truncerrors,
  :unionind,
  :unioninds,
  :unique_siteind,
  :unique_siteinds,
  :uniqueind,
  :uniqueindex,
  :uniqueinds,
  :use_debug_checks,
  :val,
  :vector,
  :δ,
  :⊕,
  :⊙,
  :⋅,
]
end
