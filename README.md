# Scheduling and production planning problems

> The optimal solution for a model is not necessarily the optimal solution for the real problem.
`["Supplement B"]`

This is a set of demo problems to try [PuLP](https://coin-or.github.io/pulp), python library for linear programming and [JuMP](https://github.com/jump-dev/JuMP.jl), Julia library for linear programming. There are three isolated tasks and a combination of them into bigger, more realistic production problem.

- [Task 1 - Production schedule for one perishable product](#task1)
- [Task 2 - Sequential production](#task2)
- [Task 3 - Select orders when demand over capacity](#task3) (mixed integer problem)
- [Task 4 - Combine tasks 1, 2 and 3](#task4)

The problems are kept simple and toy size (7 days of planning, 1-2 goods) for traceability and ease of review and testing.

Why this excercise should be useful?

- Build and test parts first before solving a bigger problem
- Differentiate model vs real world, detect what we capture in the model and what not with respect to real production,
- See if enforcing model rules can be feasible in practice
- What judgement and intuition may suggest vs the model
- What are data requirements for model to work well
- What should change on the business side (eg pricing, contract structure, sourcing procedures)
- What "lean" materially is

There are also [project notes](#Notes) and [several references](#References) below.

<a name="task1"></a>

## Task 1 - Production schedule for one perishable product 

#### Problem description

Setting:

- 1 product
- production planning for 7 days
- production quantaties are `x[t], t = 0...6`
- max production capacity C, `0 <= x[t] <= C`, let `C = 5`
- product is perishable, it can be stored for s days, let `s = 3`
- we are given a purchases schedule on each day, let `purchases = [0, 0, 2, 8, 1, 0, 1]`

Introduce target funсtion and find production volumes `x[t]`. Explore situation where result is not unique.

#### Solution formulation 

Target function:

- without target function we just find some feasible solution within the constraints
- we introduce min phycial inventories as target function to make solution unique

We had to make more assumptions:

1. closed sum - everything produced must be consumed, zero outgoing inventory
2. we set no limit on storage capacity - infinite warehouse
3. we assume full end of day clearance, all purchases made at end of day - everything produced at day t can be sold at day t

|                       | Is contrained?    |
|-----------------------|-------------------|
| Production capacity   | Yes               |
| Max storage duration  | Yes               |
| Storage capacity      | No                |
| Outgoing inventory    | Yes, set to zero  |

Other comments:

- we looked at several ways to formulate constraints for limited shelf-life / limit on storage duration  ("условие непротухания")
- minumum inventory is not necessarily the freshest shipment
- may need to check if FIFO warehouse works without goods expiry, as a safeguard for problem solution


#### Solution

Python solution code [here](task1.py).

Julia solution code [here](task1.jl).

| Day                | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|--------------------|--:|--:|--:|--:|--:|--:|--:|
| Purchases (given)  |   |   | 2 | 8 | 1 |   | 1 |
| Production (found) |   |   | 5 | 5 | 1 |   | 1 |
| Inventory          |   |   | 3 |   |   |   |   |


![](images/task1.png)

<a name="task2"></a>

## Task 2 - Sequential production, product A is a precursor to product B

- 2 products, daily production volumes `xa[t]` and `xb[t]`
- good A is precursor to good B, to produce 1 ton of B one needs 2 tons of A
- there are capacity constraints for A and B:
  - `0 <= xa[t] <= 15`
  - `0 <= xb[t] <= 5`
- there are sales orders for both A and B (`sales_a` and `sales_b`)
- goods A and B are storable, there is no shelf life constraint

Introduce target function and find optimal production of product A and B (`xa`, `xb`)

Python solution code [here](task2.py).

Julia solution code [here](task2.jl).

|               |   0 |   1 |   2 |   3 |   4 |   5 |   6 |
|:--------------|----:|----:|----:|----:|----:|----:|----:|
| **Product B**     |    |    |    |    |    |    |    |
| sales_b (given) |   1 |   0 |   0 |   7 |   4 |   0 |   1 |
| production_b  |   1 |   0 |   2 |   5 |   4 |   0 |   1 |
| inventory_b   |   0 |   0 |   2 |   0 |   0 |   0 |   0 |
| **Product A**    |    |    |    |    |    |    |    |
| processing_a  |   2 |   0 |   4 |  10 |   8 |   0 |   2 |
| sales_a (given) |   0 |   0 |   0 |  15 |   4 |   0 |   1 |
| requirement_a |   2 |   0 |   4 |  25 |  12 |   0 |   3 |
| production_a  |   2 |   0 |  14 |  15 |  12 |   0 |   3 |
| inventory_a   |   0 |   0 |  10 |   0 |   0 |   0 |   0 |

Extra notation: `processing_a + sales_a = requirement_a`

Things learned in python:

- we may omit closed sum constraint if min inventory 
- we need scale the inventory in sequential production min target function 
- we can keep only decision variables as `lpVariable`
- we can use dicts or numpy arrays

Things learned in Julia:

- we can use very compact matrix notation to declare variables and constraints
- we can use broadcasting in constants declaration

<a name="task3"></a>

## Task 3 - Select orders when demand over capacity

Pick most profitable orders from a list of orders when demand (sum of orders) is over production capacity.

Production:

- 2 products
- constant cost of production for each product, expressed as USD/t
- capacity constraints, t/day

Orders:

- total of `m` orders
- each order is (product, price, day_of_delivery)
- prices for same products may vary by order
- orders are indivisible
- choice on order is accept/reject

Simplifications:

- no storage conditions (to be relaxed)
- no inventory minimisation (to be relaxed)

#### Solution

Python solution code [here](task3.py).

Julia solution code [here](task3.jl).

Orders:

- `(order_1_quantity, order_1_price, order_1_delievery_day)`
- ...
- `(order_m_quantity, order_1_price, order_1_delievery_day)`

We will need intergers `accept_1`, ..., `accept_m`, `0 <= accept_i <= 1` to mark if order is accepted.

`purchase_t` = sum of `accept_<j> * order_quantity_<j>` for all `j` where `order_j_day == t`

Similar formulation for `profit` = `accept_<j> * order_<j>_quantity_ * (order_<j>_price - product_cost_<j>)`, 
where the only variable is `accept_<j>` and the rest are values, known at time of construction of 
target function.

Also need a good test example, showing cases when something could have gone wrong.

<a name="task4"></a>

## Task 4 - Combine tasks 1, 2 and 3

What can go wrong when combining three tasks:

- more constraints overlap
- will need more days of planning (>7)
- may need to relax more constraints (eg zero incoming and outgoing inventory)
- must write exhaustive list of assumptions
- should expiry dates affect storage at the warehouse or is is global product life?

## Notes

#### Remaining questions 

Remaining questions  about the PuLP solver:

1. The solution may be not unique, but how do we know it from solver?
   How do we extract other solutions from solver?
2. How to know which constraint was binding?
3. Are dual prices meaningful for this type of problem?
4. Is any sensitivity analysis possible?
5. Can we do multiple criteria optimisation (eg better decide on weights in target func)?
6. How to list second-best, third-best solution?
7. What would 'soft constraints' enable us to do?
8. Maybe there is a built-in nice table view of model variables?

#### Other modelling needs:

1. May need extra way to shift production right - prefer later production for any specific order.
   This happens because our product is perishable. Perhaps minimal inventory is not a guarantee 
   this is the latest-in-time production, but I may be wrong.
2. We may need a check to model FIFO stack as in [issue #1](https://github.com/epogrebnyak/linprog/issues/1)
3. We may need to model lead times for production - production may take more than 1 day,
   this will affect solutions for task 2.

## References

Software:

- [PuLP package documentation](https://coin-or.github.io/pulp)
- [Pyomo](https://www.pyomo.org)
- [JuMP](https://github.com/jump-dev/JuMP.jl)

Papers (review, etc):

- [Mixed Integer Linear Programming in Process Scheduling: Modeling, Algorithms, and Applications](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjL0N-kxL7sAhUnmYsKHWoxBm0QFjAAegQIAhAC&url=http%3A%2F%2Fcheresearch.engin.umich.edu%2Flin%2Fpublications%2FMixed%2520Integer%2520Linear%2520Programming%2520in%2520Process%2520Scheduling.pdf&usg=AOvVaw1o03XZyPw9rgAH3YG7bJOz)
- [Multiproduct, multistage machine requirements planning models](https://core.ac.uk/download/pdf/81151558.pdf)

Other:

- ["Supplement B"](http://www.uky.edu/~dsianita/300/online/LP.pdf) from [lecture notes](http://www.uky.edu/~dsianita/300/300lecture.html)
