module BenchTagSet

using BenchmarkTools
using ITensors
using LinearAlgebra

BLAS.set_num_threads(1)
ITensors.Strided.set_num_threads(1)
ITensors.enable_threaded_blocksparse(false)

suite = BenchmarkGroup()

suite["tagset"] = @benchmarkable TagSet("abcdefgh,ijklmnop,qrstuvwx,ABCDEFGH")
suite["tagset_unicode"] = @benchmarkable TagSet("αβγδϵζηθ,ijklmnop,qrstuvwx,ΑΒΓΔΕΖΗΘ")
suite["tagset_long"] = @benchmarkable TagSet(
  "abcdefghijklm,nopqrstuvwxyz,ABCDEFGHIJKLM,NOPQRSTUVWXYZ"
)
end

BenchTagSet.suite
