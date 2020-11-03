"""Task description:

- 1 product
- planning for 7 days of production
- production quantaties x[t], t = 0...6
- max production capacity C, 0 <= x[t] <= C, let C = 5
- product can be stored for s days, let s = 3
- we are given a purchases schedule purchases[t], let purchases = [0, 0, 2, 8, 1, 0, 1]
- cumulative sum of purchases is required stock of product for day t
- find x[t] (can be non-unique)

Demonstrate solver behavior under following parameters and expected results:

- all orders are under max capacity per day, no need to make stocks ps = [1, 2, 5, 2, 1, 5, 3]
- production schedule is unique, eg when ps = [0, 0, 15, 0, 0, 0, 0] and xs = [5,5,5,0, 0, 0, 0]
- production schedule not unique (as above)
- production schedule not feasisble eg ps = [0, 0, 0, 20, 0, 0, 0]
"""

using JuMP
using GLPK

function calculate(purchases, max_day_storage, max_output)
    n = length(purchases)
    model = Model(GLPK.Optimizer)
    
    # definition of the production
    @variable(model, 0 <= x[1:n] <= max_output, Int)
    
    # definition of the stored inventory
    @variable(model, 0 <= s[1:n], Int)

    # intermidiate auxiliary variables
    cp = cumsum(purchases)
    cx = cumsum(x)

    # Constraint 1: no inventory should left on the last day
    @constraint(model, s[n] == 0)

    # Constraint 2: inventory is defined as all non consumed production
    @constraint(model, [i in 1:n], cx[i] - cp[i] == s[i])

    # Constraint 3: no inventory should be spoiled, which means that inventory should not exceed demand of the next few days
    @constraint(model, [i = 1:n - max_day_storage + 1], cp[i + max_day_storage - 1] - cp[i] >= s[i])

    # Objective function
    @objective(model, Min, sum(s[i] for i in 1:n))

    optimize!(model)

    if termination_status(model) == MOI.OPTIMAL
        return (; production = Int.(value.(x)), storage = Int.(value.(s)), solved = true)
    else
        return (; production = Int[], storage = Int[], solved = false)
    end
end

production, storage, solved = calculate([0, 0, 0, 20, 0, 0, 0], 3, 5)
@assert !solved

production, storage, solved = calculate([0, 0, 15, 0, 0, 0, 0], 3, 5)
@assert solved
@assert production == [5, 5, 5, 0, 0, 0, 0]

production, storage, solved = calculate([1, 2, 5, 2, 1, 5, 3], 3, 5)
@assert solved
@assert production == [1, 2, 5, 2, 1, 5, 3]
@assert storage == [0, 0, 0, 0, 0, 0, 0]

production, storage, solved = calculate([0, 0, 2, 8, 1, 0, 1], 3, 5)
@assert solved
@assert production == [0, 0, 5, 5, 1, 0, 1]
