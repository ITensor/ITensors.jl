module BenchTagSet

using BenchmarkTools
using ITensors

suite = BenchmarkGroup()

suite["tagset"] = @benchmarkable TagSet("abcdefgh,ijklmnop,qrstuvwx,ABCDEFGH")
suite["tagset_unicode"] = @benchmarkable TagSet("αβγδϵζηθ,ijklmnop,qrstuvwx,ΑΒΓΔΕΖΗΘ")
suite["tagset_long"] = @benchmarkable TagSet("abcdefghijklm,nopqrstuvwxyz,ABCDEFGHIJKLM,NOPQRSTUVWXYZ")
end

BenchTagSet.suite
