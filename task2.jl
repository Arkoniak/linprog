"""
Task definition:
    https://github.com/epogrebnyak/linprog#task-2

Extra notation:
    
    processing_a + sales_a = requirement_a
                   sales_b = requirement_b
where                   
  processing_* - intermediate consumption
  sales_* - final client demand
  requirement_* - total demand for product

"""

using JuMP
using GLPK

# Note
# Below, you can find two versions of the `calculate` function: vector form, where each type of the production represented as a distinct vector and matrix form, where all productions are considered as a matrix of variables

function calculate(pa::AbstractVector, pb, maxa, maxb, atob)
    @assert length(pa) == length(pb)
    
    n = length(pa)
    model = Model(GLPK.Optimizer)

    @variable(model, 0 <= xa[1:n] <= maxa, Int)
    @variable(model, 0 <= xb[1:n] <= maxb, Int)
    
    reqa = pa + atob * xb
    cpa = cumsum(reqa)
    cxa = cumsum(xa)
    cpb = cumsum(pb)
    cxb = cumsum(xb)
    sa = cxa - cpa
    sb = cxb - cpb

    @constraint(model, sa .>= 0)
    @constraint(model, sb .>= 0)

    objective = sum(sa) + (atob + 1)*sum(sb)
    @objective(model, Min, objective)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return (; proda = Int.(value.(xa)),
                  stora = Int.(value.(sa)),
                  prodb = Int.(value.(xb)),
                  storb = Int.(value.(sb)),
                  reqa = Int.(value.(reqa)),
                  solved = true)
    else
        return (; proda = Int[],
                  stora = Int[],
                  prodb = Int[],
                  storb = Int[],
                  reqa = Int[],
                  solved = false)
    end
end

res = calculate([0, 0, 0, 15, 4, 0, 1], [1, 0, 0, 7, 4, 0, 1], 15, 5, 2)
@assert res.reqa == [2, 0, 4, 25, 12, 0, 3]
@assert res.proda == [2, 0, 14, 15, 12, 0, 3]
@assert res.stora == [0, 0, 10, 0, 0, 0, 0]

@assert res.prodb == [1, 0, 2, 5, 4, 0, 1]
@assert res.storb == [0, 0, 2, 0, 0, 0, 0]

########################################
# Matrix form
########################################

function calculate(p::AbstractMatrix, maxp, atob)
    @assert size(p, 2) == 2
    n, m = size(p)
    model = Model(GLPK.Optimizer)

    @variable(model, 0 <= x[1:n, j = 1:m] <= maxp[j], Int)

    req = hcat(p[:, 1] .+ atob .* x[:, 2], p[:, 2])
    cx = cumsum(x, dims = 1)
    creq = cumsum(req, dims = 1)
    s = cx .- creq

    @constraint(model, s .>= 0)

    objective = sum(s[:, 1]) + (atob + 1) * sum(s[:, 2])

    @objective(model, Min, objective)

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return (; prod = Int.(value.(x)),
                  stor = Int.(value.(s)),
                  req = Int.(value.(req)),
                  solved = true)
    else
        return (; prod = Matrix{Int}[],
                  stor = Matrix{Int}[],
                  req = Matrix{Int}[],
                  solved = false)
    end
end

res = calculate([[0, 0, 0, 15, 4, 0, 1] [1, 0, 0, 7, 4, 0, 1]], [15, 5], 2)

@assert res.solved

@assert res.req[:, 1] == [2, 0, 4, 25, 12, 0, 3]
@assert res.prod[:, 1] == [2, 0, 14, 15, 12, 0, 3]
@assert res.stor[:, 1] == [0, 0, 10, 0, 0, 0, 0]

@assert res.prod[:, 2] == [1, 0, 2, 5, 4, 0, 1]
@assert res.stor[:, 2] == [0, 0, 2, 0, 0, 0, 0]
