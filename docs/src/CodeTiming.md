
# Timing and Profiling your code

It is very important to time and profile your code to make sure your code is running as fast as possible. Here are some tips on timing and profiling your code.

If you are concerned about the performance of your code, a good place to start is Julia's [performance tips](https://docs.julialang.org/en/v1/manual/performance-tips/).

## Timing and benchmarking

Julia has many nice timing tools available. Tools like [@time](https://docs.julialang.org/en/v1/base/base/#Base.@time) and [TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl) can be used to measure the time of specific lines of code. For microbenchmarking, we recommend the [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl) package. For profiling your code, see the Julia documentation on [profiling](https://docs.julialang.org/en/v1/manual/profile/).
