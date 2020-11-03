using JuMP
using GLPK

struct Order
    day::Int
    quantity::Float64
    price::Float64
end

Order(day, quantity) = Order(day, quantity, 1)
quantity(order::Order) = order.quantity
day(order::Order) = order.day
price(order::Order) = order.price

# Given

orders = [Order(1, 1),
          Order(2, 3),
          Order(2, 2),
          Order(2, 0.5),
          Order(3, 0),
          Order(4, 5)]

function calculate(orders, max_outputs, cost_of_holding_inventory)
    n = maximum(day, orders)
    l = length(orders)

    model = Model(GLPK.Optimizer)
    @variable(model, 0 <= x[1:n] <= max_outputs)
    @variable(model, accepted[1:l], Bin)

    purchases = [quantity(orders[1]) * accepted[1]]
    for i in 2:l
        if day(orders[i]) == day(orders[i - 1])
            purchases[end] += quantity(orders[i]) * accepted[i]
        else
            push!(purchases, quantity(orders[i]) * accepted[i])
        end
    end

    cp = cumsum(purchases)
    cx = cumsum(x)
    s = cx - cp
    @constraint(model, s .>= 0)
    objective = sum(price.(orders) .* quantity.(orders) .* accepted) - sum(cost_of_holding_inventory .* s)
    @objective(model, Max, objective)

    optimize!(model)
    if termination_status(model) == MOI.OPTIMAL
        return (; accepted = Bool.(value.(accepted)),
                  storage = value.(s),
                  production = value.(x),
                  objective = objective_value(model),
                  solved = true)
    else
        return (; accepted = Bool[],
                  storage = Float64[],
                  production = Float64[],
                  objective = 0.,
                  solved = false)
    end
end

res = calculate(orders, 5, 0.1)
@assert res.solved
@assert res.storage == [0.5, 0, 0, 0]
@assert res.production == [1.5, 5, 0, 5]

res = calculate(orders, 5, 2)
@assert res.solved
@assert res.storage == [0., 0, 0, 0]
@assert res.production == [1.0, 5, 0, 5]

########################################
# Some plots
########################################
using Plots
using StableRNGs

orders = [Order(1, 1),
          Order(2, 3),
          Order(2, 2),
          Order(2, 0.5),
          Order(3, 0),
          Order(4, 5)]

x = 0:0.1:2.0
y = map(x -> calculate(orders, 5, x).objective, x)
plot(x, y)

# Generate longer and more interesting sequence
rng = StableRNG(2020)
N = 10
orders = Order[]
for i in 1:N
    L = rand(rng, 1:5)
    for j in 1:L
        q = rand(rng, 0:0.5:7)
        p = rand(rng, 0.8:0.01:1.2)
        push!(orders, Order(i, q, p))
    end
end

x = 0:0.01:5.0
y = map(x -> calculate(orders, 5, x).objective, x)
plot(x, y)

x = 1:0.1:10
y = map(x -> calculate(orders, x, 1).objective, x)
plot(x, y)
