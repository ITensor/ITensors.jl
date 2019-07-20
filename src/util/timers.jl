export printTimes,
       reset!

mutable struct Timers
  contract_t::Float64
  contract_c::Int
  gemm_t::Float64
  gemm_c::Int
  permute_t::Float64
  permute_c::Int
  svd_t::Float64
  svd_c::Int
  svd_store_t::Float64
  svd_store_c::Int
  svd_store_svd_t::Float64
  svd_store_svd_c::Int
end

global_timer = Timers()

function reset!(t::Timers)
  t.contract_t = 0.0; t.contract_c = 0;
  t.gemm_t = 0.0; t.gemm_c = 0;
  t.permute_t = 0.0; t.permute_c = 0;
  t.svd_t = 0.0; t.svd_c = 0;
  t.svd_store_t = 0.0; t.svd_store_c = 0;
  t.svd_store_svd_t = 0.0; t.svd_store_svd_c = 0;
end

function printTimes(t::Timers)
  @printf "contract_t = %.6f (%d), Total = %.4f \n" (t.contract_t/t.contract_c) t.contract_c t.contract_t
  @printf "  gemm_t = %.6f (%d), Total = %.4f \n" (t.gemm_t/t.gemm_c) t.gemm_c t.gemm_t
  @printf "  permute_t = %.6f (%d), Total = %.4f \n" (t.permute_t/t.permute_c) t.permute_c t.permute_t
  @printf "svd_t = %.6f (%d), Total = %.4f \n" (t.svd_t/t.svd_c) t.svd_c t.svd_t
  @printf "  svd_store_t = %.6f (%d), Total = %.4f \n" (t.svd_store_t/t.svd_store_c) t.svd_store_c t.svd_store_t
  @printf "  svd_store_svd_t = %.6f (%d), Total = %.4f \n" (t.svd_store_svd_t/t.svd_store_svd_c) t.svd_store_svd_c t.svd_store_svd_t
end

