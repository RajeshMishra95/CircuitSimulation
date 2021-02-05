using LinearAlgebra
using FLoops

function run(t, ncores)
    a = 0
    @floop DistributedEx(threads_basesize=t÷ncores) for i = 1:t
        p = i%2
        @reduce(a += p)
    end
    return a
end

@time begin 
    run(1000000, 2)
end