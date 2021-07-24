# Programming Language (Julia, C++) Frequently Asked Questions

## Should I use the Julia or C++ version of ITensor?

We recommend the Julia version of ITensor for most people, because:
* Julia ITensor has more and newer features than C++ ITensor, and we are developing it more rapidly
* Julia is a more productive language than C++ with more built-in features, such as linear algebra, iteration tools, etc.
* Julia is a compiled language with performance rivaling C++ (see next question below for a longer discussion)
* Julia has a rich ecosystem with a package manager, many well-designed libraries, and helpful tutorials

Even if Julia is not available by default on your computer cluster, it is easy to set up your own local install of Julia on a cluster.

However, some good reasons to use the C++ version of ITensor are:
* using ITensor within existing C++ codes
* you already have expertise in C++ programming
* multithreading support in C++, such as with OpenMP, offer certain sophisticated features compared to Julia multithreading (though Julia's support for multithreading has other benefits such as composability and is rapidly improving)
* you need other specific features of C++, such as control over memory management or instant start-up times

## Which is faster: Julia or C++ ?

Julia and C++ offer about the same performance. 

Each language gets compiled to optimized assembly code and offer arrays and containers
which can efficiently stored and iterated. Well-written Julia code can be even faster
than comparable C++ codes in many cases.

The longer answer is of course that _it depends_:
* Julia is a more productive language than C++, with many highly-optimized libraries for 
  numerical computing tasks, and excellent tools for profiling and benchmarking. 
  These features help significantly to tune Julia codes for optimal performance.
* C++ offers much more fine-grained control over memory management, which can enhance
  performance in certain applications and control memory usage.
* Julia codes can slow down significantly during refactoring or when introducing new
  code if certain [best practices](https://docs.julialang.org/en/v1/manual/performance-tips/) 
  are not followed. The most important of these is writing type-stable code. For more details
  see the [Performance Tips](https://docs.julialang.org/en/v1/manual/performance-tips/) section
  of the Julia documentation.
* C++ applications start instantly, while Julia codes can be slow to start.
  However, if this start-up time is subtracted, the rest of the time of running a 
  Julia application is similar to C++.

## Why did you choose Julia over Python for ITensor?

Julia offers much better performance than Python,
while still having nearly all of Python's benefits. One consequence is that
ITensor can be written purely in Julia, whereas to write high-performance
Python libraries it is necessary to implement many parts in C or C++ 
(the "two-language problem").

The main reasons Julia codes can easily outperform Python codes are:
1. Julia is a (just-in-time) compiled language with functions specialized
   for the types of the arguments passed to them
2. Julia arrays and containers are specialized to the types they contain, 
   and perform similarly to C or C++ arrays when all elements have the same type
3. Julia has sophisticated support for multithreading while Python has significant
   problems with multithreading

Of course there are some drawbacks of Julia compared to Python, including
a less mature ecosystem of libraries (though it is simple to call Python libraries
from Julia using [PyCall](https://github.com/JuliaPy/PyCall.jl)), and less widespread
adoption.

## Is Julia ITensor a wrapper around the C++ version?

No. The Julia version of ITensor is a complete, ground-up port
of the ITensor library to the Julia language and is written
100% in Julia.

