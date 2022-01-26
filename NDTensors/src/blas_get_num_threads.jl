# Compatible across multiple version of Julia
# (BLAS.get_num_threads() was only introduced in Julia 1.6)
blas_get_num_threads() = Compat.get_num_threads()
