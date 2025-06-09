using Pkg

# Pkg.add("Combinatorics")
# Pkg.add("DataStructures")
# Pkg.add("NPZ")

Pkg.activate(".")

using Combinatorics
using DataStructures
using Random
using NPZ
using Dates


using Logging
using Printf
using FileIO

function setup_logging()
    logdir = "TextOutputsBEAC"
    if !isdir(logdir)
        mkpath(logdir)
    end
    logfile = joinpath(logdir, "output.log")
    return open(logfile, "w")
end

function log_message(io, msg)
    println(io, msg)
    flush(io)
end

# using Statistics # for using mean()


##########################################################################################################
########################## Compute the objective value of a solution x (prices) ##########################
##########################################################################################################
# x          : x[d] = price of service din{1,…,D}
# h          : h[n,d] = h[n,d]=deterministic utility of service din{1,…,D} for customer n
#              h[n,d]=deterministic utility of the opt-out for customer n
# phi          : phi[n,s,d] = phi[n,d,s]=random utility of service din{1,…,D} for customer n under scenario s
#              phi[n,D+1,s]=deterministic utility of the opt-out for customer n under scenario s
# theta          : theta[n,s,d] = sensitivity of customer n to price of service din{1,…,D} under scenario s
##########################################################################################################
# Z          : Objective value of solution x
##########################################################################################################
function compute_obj_value(x, S, h, phi, theta)
    (NS,C) = size(theta)
    D = C-1
    idx_d = sortperm(x) # identify ordering of prices
    push!(idx_d, C) # idx_d[D+1] = D+1

    # Compute the utilities
    x = vcat(x, 0) #x[D+1]=0 (opt-out option) THIS ALLOWS TO USE 1:C, not bad
    u = zeros(NS,C)
    
    occ = zeros(Int, C)
    
    phi_  = phi
    theta_  = theta
    
    choice_matrix = Vector{Vector{Int}}()
    final_choices = zeros(Int, NS)
    
    for idx in 1:NS
        for c in 1:C
            u[idx,c] = phi_[idx,idx_d[c]]+theta_[idx,idx_d[c]]*x[idx_d[c]] # we code utilities from cheapest to highest
        end

        # to match the cooperative assumption (⟺ in case of equality, the customer
        # selects the most expensive alternative (i.e. the most useful alternative for the leader))
        for c1_c2 in collect(combinations(1:C, 2))
            c1 = c1_c2[1]
            c2 = c1_c2[2]
            #in case of equality (with tolerance to account for precision limits),
            #remove some utility to the least expensive alternative
            if(u[idx,c1] ≈ u[idx,c2])
                if(x[c1] > x[c2])
                    u[idx,c2] -= 1
                else
                    u[idx,c1] -= 1
                end
            end
        end
	end

    # Find the alternative of maximal utility for each simulated customer and compute obj. value
    y_idx = mapslices(argmax, u, dims=2)[:,:] # y_idx[n,s]=c ⟺ y[n,s,c]=1
    cnt = counter(y_idx) #cnt[c] gives the number of simulated customers ns such that y_idx=c
    Z = sum(cnt[d]*x[d] for d in 1:D)/S #value of upper-level solution x for the test sample
    
#     println("Robin cnt = ", cnt)
#     println("custom count = ", countt)
    
#     println("Sum prrofit = ", sum(prrofit) / S)
#     println("Z = ", Z)
    
    return Z, cnt
end;





##########################################################################################################
########################################### Polytime algorithm ###########################################
##########################################################################################################
# h       : h[n,d] = h[n,d]=deterministic utility of service din{1,…,D} for customer n
#           h[n,d]=deterministic utility of the opt-out for customer n
# phi       : phi[n,s,d] = phi[n,d,s]=random utility of service din{1,…,D} for customer n under scenario s
#           phi[n,D+1,s]=deterministic utility of the opt-out for customer n under scenario s
# theta       : theta[n,s,d] = sensitivity of customer n to price of service din{1,…,D} under scenario s
# a       : a[d] = lower-bound on the price of service din{1,…,D}
# b       : b[d] = upper-bound on the price of service din{1,…,D}
##########################################################################################################
# x_opt   : Optimal solution
# Z_opt/S : Optimal value (rescaled to account for the number of scenarios)
##########################################################################################################
function polytime(h, phi, theta, a, b)

    # Problem size
    (N,S,C) = size(theta)
    D = C-1
    NS = N*S

    # Flatten N×S matrices to N*S vectors and N×S×C matrices to (N*S)×C matrices
    # Result: simulated customers iin{1,…,N*S}
    phi_  = reshape(phi, NS, C)
    v_  = phi_
    theta_  = reshape(theta, NS, C)
    
#     println("")
#     println("")
#     println("phi_ = ")
#     println(phi_)
#     println("")
#     println("theta_ = ")
#     println(theta_)
#     println("")
#     println("")

    #Initial solution
    x_opt = zeros(D)
    Z_opt = 0
	
	iteration = 0
    
#     loose_index = 0
#     for i in 1:D
#         if a[i] ≈ b[i]
#             loose_index = i
#         end
#     end
    
#     sort_bnds = sortperm([a[j] for j in 1:D])

    loosis = count_loose_variables(a, b)
    
    if loosis == 1
        orders = generate_orderings(a, b)
    else
        orders = permutations(collect(1:D))
    end
    
    # Iterate through all permutations of prices orderings
    # for idx_d in permutations(collect(1:D))
    for idx_d in orders
#     for idx_d in [[1, 2, 3, 4]]
#         println("Permutation idx_d = ", idx_d) 
        # Solve the problem for x[idx_d[1]]≤x[idx_d[2]]≤…≤x[idx_d[D]]
        (x,Z) = polytime_recursion(D, N, S, v_, theta_, a, b, idx_d)
		iteration = iteration + 1
# 		println("Finished permutation  = ", iteration)
#         println("Gives x = ", x, "with obj val = ", Z)
        # Update the incumbent solution
        if(Z > Z_opt)
            x_opt = x
            Z_opt = Z
#             println("new best!")
        end
    end

    # Optimal solution
    return (x_opt, Z_opt/S)
end;


##########################################################################################################
############################## Polytime algorithm (for restricted problem) ###############################
##########################################################################################################
# D     : Number of services offered by the firm
# N     : Number of customers
# S     : Number of scenarios
# v_    : v_[i,d]=utility of service din{1,…,D} for customer i
#         v_[i,D+1]=utility of the opt-out for customer i
# theta_    : theta_[i,d]=sensitivity of customer i to price of service din{1,…,D}
# a     : a[d] = lower-bound on the price of service din{1,…,D}
# b     : b[d] = upper-bound on the price of service din{1,…,D}
# idx_d : permutation of the indexes 1,…,D. The ordering of prices x1 ≤ x2 ≤ ... ≤ x{d-1} is forced.
#--------------------------------------------------------------------------------------------------------#
# d     : next price to set
# x     : fixed prices x1 ≤ x2 ≤ ... ≤ x{d-1}
# ρ     : ρ[i] = price of the service currently selected by customer i in {1,…,N*S}
# maxU  : maxU[i] = utility of the service currently selected by customer i in {1,…,N*S} for this customer
##########################################################################################################
# x_opt : Optimal solution
# Z_opt : Optimal value
##########################################################################################################
function polytime_recursion(D, N, S, v_, theta_, a, b, idx_d; d=1, x=zeros(D), ρ=zeros(N*S), maxU=zeros(N*S))
    #################### Initialization and updates ####################
    if(d == 1) # First price
        maxU = v_[:,D+1]     # Utility of the previously selected service (initially: the opt-out (D+1))
        xd_min = a[idx_d[d]] # Lower bound on the price of service 1
    else # Other prices
        xd_min = max(a[idx_d[d]], x[idx_d[d-1]]) #lower bound on the price of service d (cannot be cheaper than the previously fixed prices)
    end
    xd_max = minimum(b[idx_d[d:end]]) # Upper bound on the price of service d (cannot be more expensive than the UB on the following prices)

    # Infeasible solution (does not respect x[idx_d[1]]≤x[idx_d[2]]≤x[idx_d[D]]), stop the recursion
    if(xd_min > xd_max)
        return (x,0)
    end

    # Current optimal solution and value
    x_opt = deepcopy(x)
    Z_opt = sum(ρ)

    ################### General case: prices 1 to D-1 (O(n) operations in the innermost loop, + recursion) ###################
    if(d ≤ D-1)
        cp = (maxU - v_[:,idx_d[d]])./theta_[:,idx_d[d]] # Critical prices: service d is preferred to all the cheaper services by customer n ⟺ x[d] ≤ cp[n]
                                                         # Sure! if it gets too expensive, its utility will be too low and we dont get any profit from d
        
        # xd = x[d], xd_max = upperbound on price x[d]
        # when the upper bound, thus all possible prices in the range, are < cp, then new alt d is always better. thus sure capture
        idx_sure = findall(xd_max .≤ cp )            # Indexes of the customers that will be captured by service d, no matter its price
        idx_poss = findall(xd_min .≤ cp .< xd_max)   # Indexes of the customers that can be captured by service d, depending on its price
        cp_poss = cp[idx_poss]                       # Critical price of the "possible" customers
        sorted_idx = sortperm(cp_poss)               # Sort possible customers with respect to their critical price
        idx_poss_sorted = idx_poss[sorted_idx]       # Indexes of the possible customers, sorted with respect to their critical price
        ρ_new = deepcopy(ρ)                          # New profit obtained from each customer
        maxU_new = deepcopy(maxU)                    # Utility of the selected service for each customer
        
#         println("n_sure = ", length(idx_sure))

        # I) Initialize the price of service d to its upper bound
        x[idx_d[d]] = xd_max                                                      # New price for service d
        ρ_new[idx_sure] .= xd_max                                                 # New profit for customers selecting service d at this price point
        maxU_new[idx_sure] = v_[idx_sure,idx_d[d]] + xd_max*theta_[idx_sure,idx_d[d]] # New maximum utility for these customers
        # its new max for the sure ones, because no matter which price of d we said they are going to choose it, so it is the max util
        (x_rec, Z_rec) = polytime_recursion(D, N, S, v_, theta_, a, b, idx_d; d=d+1, x=x, ρ=ρ_new, maxU=maxU_new) # Recursively solve the restricted problem given that x1,x2,...,xd are fixed
        if(Z_rec > Z_opt) # Update the incumbent solution
            Z_opt = Z_rec
            x_opt = deepcopy(x_rec)
        end
        
        # this will initialize ALL prices to their upperbounds, because it starts at 1, but one does not continue before it did the same for 2
        # and then for 3 and 4 etc until J

        # II) Test all the critical prices of service d (for possible customers)
        # sorted_idx = sorted indices of all customers that MIGHT switch to d. 
        # the last index of those, i.e. index = length, is the customer that switches away from d the last somehow, because it will be at the highest price out of everyone
        last_idx_switched = length(sorted_idx) # Index of the last customer to switch from its previous service to d (initially, only the one with the highest critical price)
        # i feel like this should say "to switch from d to its previous service" because they have highest breakpoint
        
        # you start at the highest critical point, because there you know that everyone else, that critical point lower than that
        # the will NOT choose d
        # except for the ones that switch, i.e. the idx_switched. This will first be one, then two etc as 
        # we climb from top of the bps down.
        
        for idx in reverse(sorted_idx) # Test all critical prices, starting from the highest one
            xd = cp_poss[idx] # New price of service d
            x[idx_d[d]] = xd
            idx_poss_switched = idx_poss_sorted[last_idx_switched:end] # Possible customers switching to d at this price point
            ρ_new[idx_sure] .= xd          #Update profits (sure customers)
            ρ_new[idx_poss_switched] .= xd #Update profits (possible customers)
            maxU_new[idx_sure] = v_[idx_sure,idx_d[d]] + xd*theta_[idx_sure,idx_d[d]] #Update max utilities (sure customers)
            maxU_new[idx_poss_switched] = v_[idx_poss_switched,idx_d[d]] + xd*theta_[idx_poss_switched,idx_d[d]] #Update max utilities (possible customers)
            (x_current, Z_current) = polytime_recursion(D, N, S, v_, theta_, a, b, idx_d; d=d+1, x=x, ρ=ρ_new, maxU=maxU_new) # Recursively solve the restricted problem given that x1,x2,...,xd are fixed
            
#             if Z_current != 0
# #                 println("xd = ", xd, " gives Z = ", Z_current)            
#                 if xd == 0.5694060046407339 || xd == 0.569386111407083
#                     n_sure = length(idx_sure) 
#                     n_switched = length(idx_poss_switched)
#                     println("xd = ", xd)
#                     println("n_sure = ", n_sure)
#                     println("n_switched = ", n_switched)
# #                     printlnt("Z_lost before = ", Z_lost - ρ[idx_poss[idx]])
# #                     printlnt("Z_lost now = ", Z_lost)
#                     println("person switched from $(idx_poss[idx]) which had profit $(ρ[idx_poss[idx]])")
# #                     pritnln("Z_init = ", Z_init)
#                     println("(n_sure+n_switched)*xd = ", (n_sure+n_switched)*xd)
#                     println("Z_current = ", Z_current)
#                 end
#             end
            
            if(Z_current > Z_opt) # Update the incumbent solution
                Z_opt = Z_current
                x_opt = deepcopy(x_current)
#                 println("new best")
            end
            last_idx_switched -= 1 # One more customer will switch to service d at the next iteration
        end

    ################### Terminal case: prices D (O(1) operations in the innermost loop) ###################
    else
        cp = (maxU - v_[:,idx_d[d]])./theta_[:,idx_d[d]] # Critical prices: service d is preferred to all the cheaper services by customer n ⟺ x[d] ≤ cp[n]
        idx_sure = findall(xd_max .≤ cp)             # Indexes of the customers that will be captured by service d, no matter its price
        n_sure = length(idx_sure)                    # Number of "sure" customers
        idx_poss = findall(xd_min .≤ cp .< xd_max)   # Indexes of the customers that can be captured by service d, depending on its price
        cp_poss = cp[idx_poss]                       # Critical price of the "possible" customers
        sorted_idx = sortperm(cp_poss)               # Sort possible customers with respect to their critical price
        n_switched = 0                               # Number of possible customers currently switched to service d (initially 0)
        Z_init = Z_opt - sum(ρ[idx_sure])            # Initial value: We remove the previous profits from customers that will switch
        Z_lost = 0                                   # Profit lost on possible customers due to them abandoning their previous service

        # I) Initialize the price of service d to its upper bound
        x_opt[idx_d[d]] = xd_max       # New price of service d
        Z_opt = Z_init + n_sure*xd_max # Only the sure customers switch, leading to a profit of n_sure*xd_max
        
        # II) Test all the critical prices of service d (for possible customers)
        for idx in reverse(sorted_idx) # Test all critical prices, starting from the highest one
            xd = cp_poss[idx]                             # New price of service d
            x[idx_d[d]] = xd
            n_switched += 1                              # One more customer switches to service d
            Z_lost += ρ[idx_poss[idx]]                   # We loose the profit of its previously selected service
            Z = Z_init + (n_sure+n_switched)*xd - Z_lost # New value: n_sure+n_switched customers now select service d
#             println("deep level: x = ", x, " gives Z = ", Z)
            if(Z > Z_opt) # Update the incumbent solution
                Z_opt = Z
                x_opt = deepcopy(x)
#                 println("deep level: new best")
            end
        end
    end

    # Degenerate case x_opt[d]=0 indicates that d is never selected in the optimal solution. Set its price to its upper bound
    x_opt = x_opt + b.*(x_opt.==0)

    # Return the optimal solution for the current restricted problem
    return (x_opt, Z_opt)
end;

function generate_orderings(a::Vector, b::Vector)
    # Check if vectors a and b are of the same length
    if length(a) != length(b)
        error("Vectors a and b must be of the same length")
    end

    # Identify the loose variable
    loose_idx = findfirst(x -> a[x] != b[x], 1:length(a))
    if loose_idx === nothing
        error("No loose variable found")
    end

    # Extract the bounds for the fixed variables and sort them
    fixed_indices = setdiff(1:length(a), [loose_idx])
    fixed_bounds = [(idx, a[idx]) for idx in fixed_indices]
    sort!(fixed_bounds, by = x -> x[2])

    # Generate all possible orderings
    orderings = []
    for pos in 1:length(fixed_bounds) + 1
        ordering = [x[1] for x in fixed_bounds]
        insert!(ordering, pos, loose_idx)
        # Check if the ordering is valid
        if isvalidordering(a, b, ordering)
            push!(orderings, ordering)
        end
    end

    return orderings
end

function isvalidordering(a::Vector, b::Vector, ordering::Vector)
    for i in 1:length(ordering)-1
        if a[ordering[i]] > b[ordering[i+1]]
            return false
        end
    end
    return true
end;

function count_loose_variables(a::Vector, b::Vector)
    # Check if vectors a and b are of the same length
    if length(a) != length(b)
        error("Vectors a and b must be of the same length")
    end

    # Count the number of loose variables
    loose_count = count(i -> a[i] != b[i], 1:length(a))

    return loose_count
end;

function compute_obj_value_priority_queue(x, h, phi, theta, caps, prio_queue)
    (NS,C) = size(theta)
    D = C-1
    idx_d = sortperm(x) # identify ordering of prices
    push!(idx_d, C) # idx_d[D+1] = D+1
    push!(caps, NS+1)

    # Compute the utilities
    x = vcat(x, 0) #x[D+1]=0 (opt-out option) THIS ALLOWS TO USE 1:C, not bad
    u = zeros(NS,C)
    
    occ = zeros(Int, C)
    
    phi_  = phi
    theta_  = theta
    
    final_choices = zeros(Int, NS)
    
    for idx in prio_queue
        for c in 1:C
            u[idx,c] = phi_[idx,c]+theta_[idx,c]*x[c]
        end

        # to match the cooperative assumption (⟺ in case of equality, the customer
        # selects the most expensive alternative (i.e. the most useful alternative for the leader))
        for c1_c2 in collect(combinations(1:C, 2))
            c1 = c1_c2[1]
            c2 = c1_c2[2]
            #in case of equality (with tolerance to account for precision limits),
            #remove some utility to the least expensive alternative
            if(u[idx,c1] ≈ u[idx,c2])
                if(x[c1] > x[c2])
                    u[idx,c2] -= 1
                else
                    u[idx,c1] -= 1
                end
            end
        end

        utils = [u[idx,j] for j in 1:C]
        ordering = sortperm(utils, rev=true)

        # now just go through their preferences, and if there is space we put them there, if not,
        # they take the next best one
        assigned = false
        j = 1
        while j <= C && !assigned
            if occ[ordering[j]] <= caps[ordering[j]] - 1
                final_choices[idx] = ordering[j]
                occ[ordering[j]] += 1
                assigned = true
            else
                j += 1
            end 
        end
    end
    
    Z = sum(occ[j] * x[j] for j in 1:D)
    
    return Z, occ, final_choices
end;

function compute_obj_value_with_forced_capacities(x, h, phi, theta, caps; inv=false)
    (NS,C) = size(theta)
    D = C-1
    idx_d = sortperm(x) # identify ordering of prices
    push!(idx_d, C) # idx_d[D+1] = D+1

    # Compute the utilities
    x = vcat(x, 0) #x[D+1]=0 (opt-out option) THIS ALLOWS TO USE 1:C, not bad
    u = zeros(NS,C)
    
    occ = zeros(Int, C)
    
    phi_  = phi
    theta_  = theta
    
    choice_matrix = Vector{Vector{Int}}()
    final_choices = zeros(Int, NS)
    
    for idx in 1:NS
        for c in 1:C
            u[idx,c] = phi_[idx,idx_d[c]]+theta_[idx,idx_d[c]]*x[idx_d[c]] # we code utilities from cheapest to highest
        end

        # to match the cooperative assumption (⟺ in case of equality, the customer
        # selects the most expensive alternative (i.e. the most useful alternative for the leader))
        for c1_c2 in collect(combinations(1:C, 2))
            c1 = c1_c2[1]
            c2 = c1_c2[2]
            #in case of equality (with tolerance to account for precision limits),
            #remove some utility to the least expensive alternative
            if(u[idx,c1] ≈ u[idx,c2])
                if(x[c1] > x[c2])
                    u[idx,c2] -= 1
                else
                    u[idx,c1] -= 1
                end
            end
        end

        utils = [u[idx,j] for j in 1:C]
        ordering = sortperm(utils, rev=true)
        replace!(ordering, D+1 => 0) # replace D+1 by 0 in order to facilitate ordering
        push!(choice_matrix, ordering) # add the ordering to the matrix
    end
    
    if !inv
        sorted_orderings = sort(choice_matrix, lt=isless)
    else
        sorted_orderings = sort(choice_matrix, lt=isless, rev=true)
    end
    
    while length(sorted_orderings) >= 1
        # check next person in line's first preference and remove them from set
        next_pref = popfirst!(sorted_orderings)[1]
        # add them to that alternative
        if next_pref >= 1 # if its not the opt-out
            occ[idx_d[next_pref]] += 1 # add them to correct choice
            if occ[idx_d[next_pref]] == caps[idx_d[next_pref]] # if we reached capacity for that choice
                # delete it from all rows in the matrix. but not the choice it corresponds to, rather its pricerank
                for i in 1:length(sorted_orderings)
                    filter!(x -> x != next_pref, sorted_orderings[i])
                end
                # reorder the matrix lexicographically 
                if !inv
                    sorted_orderings = sort(sorted_orderings, lt=isless)
                else
                    sorted_orderings = sort(sorted_orderings, lt=isless, rev=true)
                end
                
            end
        end
    end
    
    Z = sum(occ[j] * x[j] for j in 1:D)
    
    return Z, occ, final_choices # I actually dont compute the final choices, sorry
end;

# Function to recursively search through all combinations
function search(phi, theta, current_price, lowerbounds, upperbounds, step_size, dimension, caps, prio_queue;inv=false)
    global optimal_price, max_profit
    D = length(current_price)
    
    if dimension > D
        if isnothing(caps)
			current_profit, _ = compute_obj_value(current_price, S, zeros(D,D+1), phi, theta)
        else
            if isnothing(prio_queue) 
				if !inv 
                	current_profit, _, _ = compute_obj_value_with_forced_capacities(current_price, zeros(D,D+1), phi, theta, caps)
				else
					current_profit, _, _ = compute_obj_value_with_forced_capacities(current_price, zeros(D,D+1), phi, theta, caps;inv)
				end
            else
                current_profit, _, _ = compute_obj_value_priority_queue(current_price, zeros(D,D+1), phi, theta, caps, prio_queue)
            end
        end
        if current_profit > max_profit
            max_profit = current_profit
            optimal_price = copy(current_price)
        end
        return
    end

    for price in lowerbounds[dimension]:step_size:upperbounds[dimension]
        current_price[dimension] = price
        search(phi, theta, current_price, lowerbounds, upperbounds, step_size, dimension + 1, caps, prio_queue;inv)
    end
end

function brute_force(h, phi, theta, a, b, step_size, caps=nothing, prio_queue=nothing; inv=false)
    global optimal_price, max_profit
    # Define your bounds and step size
    (N,S,C) = size(theta)
    D = C-1
    lowerbounds = a
    upperbounds = b
    step_size = 0.01 # Example value
	NS = N*S
    phi_  = reshape(phi, NS, C)
    theta_  = reshape(theta, NS, C)
    # Initialize variables for the optimal price and maximum profit
    optimal_price = copy(lowerbounds)
    max_profit = -Inf

    # Run the search
    search(phi_, theta_, zeros(D), lowerbounds, upperbounds, step_size, 1, caps, prio_queue;inv)
    if !isnothing(caps)
        max_profit = max_profit/S
    end
    return optimal_price, max_profit
end;

##########################################################################################################
################################## Simplified Polytime algorithm #########################################
##########################################################################################################
# h       : h[n,d] = h[n,d]=deterministic utility of service din{1,…,D} for customer n
#           h[n,d]=deterministic utility of the opt-out for customer n
# phi       : phi[n,s,d] = phi[n,d,s]=random utility of service din{1,…,D} for customer n under scenario s
#           phi[n,D+1,s]=deterministic utility of the opt-out for customer n under scenario s
# theta       : theta[n,s,d] = sensitivity of customer n to price of service din{1,…,D} under scenario s
# a       : a[d] = lower-bound on the price of service din{1,…,D}
# b       : b[d] = upper-bound on the price of service din{1,…,D}
##########################################################################################################
# x_opt   : Optimal solution
# Z_opt/S : Optimal value (rescaled to account for the number of scenarios)
##########################################################################################################
function polytime_simple_multibreak_caps(h, phi, theta, a, b, caps, prio_queue=nothing;inv=false)
	log_io = setup_logging()
	start_time = now()
	
    # Problem size
    (N,S,C) = size(theta)
    D = C-1
    NS = N*S

    # Flatten N×S matrices to N*S vectors and N×S×C matrices to (N*S)×C matrices
    # Result: simulated customers iin{1,…,N*S}
    phi_  = reshape(phi, NS, C)
    v_  = phi_
    theta_  = reshape(theta, NS, C)

    #Initial solution
    x_opt = zeros(D)
    Z_opt = 0
	
	x_opt = (a + b) / 2
	Z_opt, _ = compute_obj_value_priority_queue(x_opt, zeros(N,D+1), v_, theta_, caps, prio_queue)
	println("first prices = ", x_opt)
	println("obj value = ", Z_opt)
	flush(stdout)
	
	log_message(log_io, "first prices = $(x_opt)")
	log_message(log_io, "obj value = $(Z_opt)")
	
	iteration = 0
    
    loosis = count_loose_variables(a, b)
    
    if loosis == 1
        orders = generate_orderings(a, b)
    else
        orders = permutations(collect(1:D))
    end
    
    # Iterate through all permutations of prices orderings
    # for idx_d in permutations(collect(1:D))
    for idx_d in orders
        (x,Z) = polytime_recursion_simple_multibreak_caps(D, N, S, v_, theta_, a, b, idx_d, caps, prio_queue;inv)
		elapsed_time = now() - start_time
		println("new ordering solved after ", elapsed_time)
		println("prices = ", x)
		println("obj value = ", Z)
		flush(stdout)
		
		log_message(log_io, "new ordering solved after $(elapsed_time)")
		log_message(log_io, "prices = $(x)")
		log_message(log_io, "obj value = $(Z)")
		
        iteration = iteration + 1
        if(Z > Z_opt)
            x_opt = x
            Z_opt = Z
			elapsed_time = now() - start_time
			println("new best after ", elapsed_time)
			println("prices = ", x_opt)
			println("obj value = ", Z_opt)
			println("")
			flush(stdout)
			
			log_message(log_io, "new best after $(elapsed_time)")
			log_message(log_io, "prices = $(x_opt)")
			log_message(log_io, "obj value = $(Z_opt)")
			log_message(log_io, "")
        end
    end
	close(log_io)
    # Optimal solution
    return (x_opt, Z_opt/S)
end;


##########################################################################################################
############################## Polytime algorithm (for restricted problem) ###############################
##########################################################################################################
# D     : Number of services offered by the firm
# N     : Number of customers
# S     : Number of scenarios
# v_    : v_[i,d]=utility of service din{1,…,D} for customer i
#         v_[i,D+1]=utility of the opt-out for customer i
# theta_    : theta_[i,d]=sensitivity of customer i to price of service din{1,…,D}
# a     : a[d] = lower-bound on the price of service din{1,…,D}
# b     : b[d] = upper-bound on the price of service din{1,…,D}
# idx_d : permutation of the indexes 1,…,D. The ordering of prices x1 ≤ x2 ≤ ... ≤ x{d-1} is forced.
#--------------------------------------------------------------------------------------------------------#
# d     : next price to set
# x     : fixed prices x1 ≤ x2 ≤ ... ≤ x{d-1}
# ρ     : ρ[i] = price of the service currently selected by customer i in {1,…,N*S}
# maxU  : maxU[i] = utility of the service currently selected by customer i in {1,…,N*S} for this customer
##########################################################################################################
# x_opt : Optimal solution
# Z_opt : Optimal value
##########################################################################################################
function polytime_recursion_simple_multibreak_caps(D, N, S, v_, theta_, a, b, idx_d, caps, prio_queue; d=1, x=zeros(D),inv=false)
    #################### Initialization and updates ####################
    if(d == 1) # First price
        xd_min = a[idx_d[d]] # Lower bound on the price of service 1
    else # Other prices
        xd_min = max(a[idx_d[d]], x[idx_d[d-1]]) #lower bound on the price of service d (cannot be cheaper than the previously fixed prices)
    end
    xd_max = minimum(b[idx_d[d:end]]) # Upper bound on the price of service d (cannot be more expensive than the UB on the following prices)

    # Infeasible solution (does not respect x[idx_d[1]]≤x[idx_d[2]]≤x[idx_d[D]]), stop the recursion
    if(xd_min > xd_max)
        return (x,0)
    end

    # Current optimal solution and value
    x_opt = deepcopy(x)
    Z_opt = 0

    ################### General case: prices 1 to D-1 (O(n) operations in the innermost loop, + recursion) ###################
    if(d ≤ D-1)
        # I) Initialize the price of service d to its upper bound
        x[idx_d[d]] = xd_max                                                      # New price for service d
        
        # compute ALL breakpoints
        cp = Float64[]
        for idx in 1:(N*S)
            # compute the utilities of all alternatives except idx_d[d]
            #utils = [v_[idx,k] + theta_[idx,k] * x[k] for k in 1:D if k != idx_d[d]] 
            
            # so technically you only need to do the ones introduced so far
            utils = [v_[idx,idx_d[k]] + theta_[idx,idx_d[k]] * x[idx_d[k]] for k in 1:D if k < d] 
            
            push!(utils, v_[idx,D+1]) # add opt out

            for l in 1:length(utils)
                bp = (utils[l] - v_[idx,idx_d[d]]) / theta_[idx,idx_d[d]]
#                 if bp >= a[idx_d[d]] && bp <= b[idx_d[d]]
#                     push!(cp, bp)
#                 end
                push!(cp, bp)
            end
        end
        
        J = D
        
        # Filter all bp values outside the bounds
        cp = filter(x -> x >= a[idx_d[d]] && x <= b[idx_d[d]], cp)
		
		# dont forget to add the upper and lower bound to the breakpoints!
		push!(cp, b[idx_d[d]])
		push!(cp, a[idx_d[d]])
        
        # println("Officially computed nr of bps = $countt")
        # println("J*N*S = $(J*N*S)")
        # println("Length of cp = $(length(cp))")
        
        for xd in cp 
            x[idx_d[d]] = xd
            (x_current, Z_current) = polytime_recursion_simple_multibreak_caps(D, N, S, v_, theta_, a, b, idx_d, caps, prio_queue; d=d+1, x=x,inv=inv) # Recursively solve the restricted problem given that x1,x2,...,xd are fixed
            if(Z_current > Z_opt) # Update the incumbent solution
                Z_opt = Z_current
                x_opt = deepcopy(x_current)
            end
        end

    ################### Terminal case: prices D (O(1) operations in the innermost loop) ###################
    else
        # compute ALL breakpoints
        cp = Float64[]
        for idx in 1:(N*S)
            # compute the utilities of all alternatives except idx_d[d]
#             utils = [v_[idx,k] + theta_[idx,k] * x[k] for k in 1:D if k != idx_d[d]] 
            
            # so technically you only need to do the ones introduced so far
            utils = [v_[idx,idx_d[k]] + theta_[idx,idx_d[k]] * x[idx_d[k]] for k in 1:D if k < d] 
            
            push!(utils, v_[idx,D+1])

            for l in 1:length(utils)
                bp = (utils[l] - v_[idx,idx_d[d]]) / theta_[idx,idx_d[d]]
#                 if bp >= a[idx_d[d]] && bp <= b[idx_d[d]]
#                     push!(cp, bp)
#                 end
                push!(cp, bp)
            end
        end
        
        # Filter all bp values outside the bounds
        cp = filter(x -> x >= a[idx_d[d]] && x <= b[idx_d[d]], cp)
		
		# dont forget to add the upper and lower bound to the breakpoints!
		push!(cp, b[idx_d[d]])
		push!(cp, a[idx_d[d]])
        
        J = D
        
        # println("Officially computed nr of bps = $countt")
        # println("J*N*S = $(J*N*S)")
        # println("Length of cp = $(length(cp))")
        
        # I) Initialize the price of service d to its upper bound
        x_opt[idx_d[d]] = xd_max       # New price of service d

        # II) Test all the critical prices of service d (for possible customers)
        for xd in cp                          
            x[idx_d[d]] = xd
            # maybe we can have a separate one that only cares about eh computing Z? 
            # Although computing the choices is very elemental to the compute function I think
            if isnothing(prio_queue) 
                if !inv
                    Z, _, _ = compute_obj_value_with_forced_capacities(x, zeros(N,D+1), v_, theta_, caps)
                else
                    Z, _, _ = compute_obj_value_with_forced_capacities(x, zeros(N,D+1), v_, theta_, caps; inv)
                end
            else
                Z, _, _ = compute_obj_value_priority_queue(x, zeros(N,D+1), v_, theta_, caps, prio_queue)
            end
            
            if(Z > Z_opt) # Update the incumbent solution
                Z_opt = Z
                x_opt = deepcopy(x)
            end
        end
    end

    # Degenerate case x_opt[d]=0 indicates that d is never selected in the optimal solution. Set its price to its upper bound
    x_opt = x_opt + b.*(x_opt.==0)

    # Return the optimal solution for the current restricted problem
    return (x_opt, Z_opt)
end;



function polytime_onep(h, phi, theta, a, b, ρ)
    # Problem size
    (N,S,C) = size(theta)
    D = C-1
    NS = N*S

    # Flatten N×S matrices to N*S vectors and N×S×C matrices to (N*S)×C matrices
    # Result: simulated customers iin{1,…,N*S}
    phi_  = reshape(phi, NS, C)
    v_  = phi_
    theta_  = reshape(theta, NS, C)
    ρ_ = reshape(ρ, NS)

    #Initial solution
    x_opt = zeros(D)
    Z_opt = 0
	iteration = 0

    loosis = count_loose_variables(a, b)
    
    if loosis == 1
        orders = generate_orderings(a, b)
    else
        orders = permutations(collect(1:D))
    end
    
    # Iterate through all permutations of prices orderings
    # for idx_d in permutations(collect(1:D))
    for idx_d in orders
        # Solve the problem for x[idx_d[1]]≤x[idx_d[2]]≤…≤x[idx_d[D]]
        (x,Z) = polytime_recursion_onep(D, N, S, v_, theta_, a, b, idx_d, ρ=ρ_)
		iteration = iteration + 1
        if(Z > Z_opt)
            x_opt = x
            Z_opt = Z
        end
    end

    # Optimal solution
    return (x_opt, Z_opt/S)
end;

function polytime_recursion_onep(D, N, S, v_, theta_, a, b, idx_d; d=1, x=zeros(D), ρ=zeros(N*S), maxU=zeros(N*S))
    #################### Initialization and updates ####################
    maxU = v_[:,D+1]  # Utility of the previously selected service (i.e.: the opt-out (D+1))
    xd_min = a[idx_d[d]] # Lower bound on the price of service 1
    xd_max = minimum(b[idx_d[d:end]]) # Upper bound on the price of service d (cannot be more expensive than the UB on the following prices)
    
    # Infeasible solution (does not respect x[idx_d[1]]≤x[idx_d[2]]≤x[idx_d[D]]), stop the recursion
    if(xd_min > xd_max)
        return (x,0)
    end

    # Current optimal solution and value
    x_opt = deepcopy(x)
    Z_opt = sum(ρ)
    
    ################### Terminal case: prices D (O(1) operations in the innermost loop) ###################
    cp = (maxU - v_[:,idx_d[d]])./theta_[:,idx_d[d]] # Critical prices: service d is preferred to all the cheaper services by customer n ⟺ x[d] ≤ cp[n]
    idx_sure = findall(xd_max .≤ cp)             # Indexes of the customers that will be captured by service d, no matter its price
    n_sure = length(idx_sure)                    # Number of "sure" customers
    idx_poss = findall(xd_min .≤ cp .< xd_max)   # Indexes of the customers that can be captured by service d, depending on its price
    cp_poss = cp[idx_poss]                       # Critical price of the "possible" customers
    sorted_idx = sortperm(cp_poss)               # Sort possible customers with respect to their critical price
    n_switched = 0                               # Number of possible customers currently switched to service d (initially 0)
    Z_init = Z_opt - sum(ρ[idx_sure])            # Initial value: We remove the previous profits from customers that will switch
    Z_lost = 0                                   # Profit lost on possible customers due to them abandoning their previous service
    
    # I) Initialize the price of service d to its upper bound
    x_opt[idx_d[d]] = xd_max       # New price of service d
    Z_opt = Z_init + n_sure*xd_max # Only the sure customers switch, leading to a profit of n_sure*xd_max
    shadow_switches = []
    
    for idx in reverse(sorted_idx) # Test all critical prices, starting from the highest one
        xd = cp_poss[idx]                             # New price of service d
         
        if xd < ρ[idx_poss[idx]]                      # if the previous profit is bigger the customer does
            push!(shadow_switches, [idx, xd, ρ[idx_poss[idx]]])  # not switch, but he will at the next lowest 
            continue                                             # breakpoint
        end
        
        if !isempty(shadow_switches)                  # check if there are breakpoints in the queue
            if xd < shadow_switches[1][2]             # if there are, and the current bp is smaller than 
                slen = length(shadow_switches)
                n_switched += slen                    # those in the queue, add these customers to n_switched
                Z_lost += sum(sw[3] for sw in shadow_switches)  # and substract all the lost profit
                empty!(shadow_switches)                         # and empty the queue
            end
        end
        
        n_switched += 1                              # One more customer switches to service d
        x[idx_d[d]] = xd                             
        Z_lost += ρ[idx_poss[idx]]                   # We loose the profit of its previously selected service
        Z = Z_init + (n_sure+n_switched)*xd - Z_lost # New value: n_sure+n_switched customers now select service d
            
        if(Z > Z_opt) # Update the incumbent solution
            Z_opt = Z
            x_opt = deepcopy(x)
        end
    end

    # Degenerate case x_opt[d]=0 indicates that d is never selected in the optimal solution. Set its price to its upper bound
    x_opt = x_opt + b.*(x_opt.==0)

    # Return the optimal solution for the current restricted problem
    return (x_opt, Z_opt)
end;

function do_poly_one_price_alt(exo_utility, endo_coef, prices, loose_index, a, b)
    N = size(exo_utility)[1]
    S = size(exo_utility)[2]
    D = size(exo_utility)[3] - 1
    h = zeros(N, D+1)
    a_ = copy(prices)
    b_ = copy(prices)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]
    (x_poly, Z_poly) = polytime(h, exo_utility, endo_coef, a_, b_)
    return (x_poly[loose_index], Z_poly)
end;

function reduce_to_one_price_case(exo_utility, endo_coef, prices, loose_index)
    N = size(exo_utility)[1]
    R = size(exo_utility)[2]
    D = size(prices)[1]
    exo_utility_new = zeros(N, R, 2)
    endo_coef_new = zeros(N, R, 2)
    ρ = zeros(N, R)
    new_prices = vcat(prices, 0) # add profit of 0 in case the opt-out has highest utility
    choices = zeros(N,R)
    
    # Loop over n and r
    for n in 1:N
        for r in 1:R
            # Compute the utility for the D+1 value... which is the opt-out.
            utility_Dplus1 = exo_utility[n, r, D + 1]
            utility_loose = exo_utility[n, r, loose_index] + endo_coef[n, r, loose_index] * new_prices[loose_index]

            # Initialize variables to store the maximum utility and its index for this 
            # combination of n and r
            max_util_n_r = utility_Dplus1
            max_index_n_r = D + 1

            # Loop over i from 1 to D, excluding the index loose_index
            for i in 1:D
                if i == loose_index
                    continue  # Skip this iteration if i is != loose_index
                end
                utility_i = exo_utility[n, r, i] + endo_coef[n, r, i] * new_prices[i]
                
                # break draws
                if utility_i ≈ max_util_n_r
                    if new_prices[i] < new_prices[max_index_n_r]
                        utility_i -= 1 
                    else
                        max_util_n_r -= 1
                    end
                end
                
                if (utility_i > max_util_n_r)
                    max_util_n_r = utility_i
                    max_index_n_r = i
                end
                
            end
            
            exo_utility_new[n, r, 1] = exo_utility[n, r, loose_index]
            endo_coef_new[n, r, 1] = endo_coef[n, r, loose_index]
            exo_utility_new[n, r, 2] = max_util_n_r
            endo_coef_new[n, r, 2] = 0
            ρ[n, r] = new_prices[max_index_n_r]
            choices[n, r] = max_index_n_r
        end
    end
    return (exo_utility_new, endo_coef_new, ρ, choices)
end;

function do_poly_one_price(exo_utility, endo_coef, prices, loose_index, a, b)
    exo_utility_new, endo_coef_new, ρ, choices = reduce_to_one_price_case(exo_utility, endo_coef, prices, loose_index)
    N = size(exo_utility)[1]
    S = size(exo_utility)[2]
    D = size(exo_utility)[3] - 1
    h = zeros(N,2)
    (x_poly, Z_poly) = polytime_onep(h, exo_utility_new, endo_coef_new, a, b, ρ)
    return (x_poly[1], Z_poly)
end;

function polytime_heuristic(exo_utility, endo_coef, start_prices, a, b)
    # initialize prices
    D = size(exo_utility)[3] - 1
    start_prices = Float64.(start_prices) # cast prices to float in case the start prices were all integer
    prices = copy(start_prices)
    
    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1
    best_obj = 0
    
    best_prices = copy(prices)
    consecutive_close = 0
    old_obj = 0
    
    # Define a threshold for convergence
    epsilon = 1e-9
    iter = 0 # failsafe to not run forever in case of non-convergence
    
    obj_val = best_obj
    
    while consecutive_close < D && iter < 20
        iter += 1       
        
        # Call do_poly_one_price to get x_poly and obj_val
        #loose_price, obj_val = do_poly_one_price(exo_utility, endo_coef, prices, loose_index, [a[loose_index]], [b[loose_index]])
        loose_price, obj_val = do_poly_one_price(exo_utility, endo_coef, prices, loose_index, [a[loose_index]], [b[loose_index]])
        
        # Update the price at the current index
        prices[loose_index] = loose_price

        if obj_val > best_obj
            best_obj = obj_val
            best_prices = copy(prices)
        end
        
        # Check if the current obj_val is very close to the previous one
        if abs(obj_val - old_obj) < epsilon
            consecutive_close += 1 # Increment the counter
        else
            consecutive_close = 0  # Reset consecutive_close counter
        end
        
        old_obj = obj_val
        
        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > D
            loose_index = 1
        end
    end
    
    return best_obj, best_prices, iter
end;

function polytime_simple_multibreak_caps_simpler(phi, theta, S, a, b, caps, prio_queue=nothing; inv=false)

    # Problem size
    (NS,C) = size(theta)
    D = C-1

    # Flatten N×S matrices to N*S vectors and N×S×C matrices to (N*S)×C matrices
    # Result: simulated customers iin{1,…,N*S}
    phi_  = phi
    v_  = phi_
    theta_  = theta

    #Initial solution
    x_opt = zeros(D)
    Z_opt = 0
	iteration = 0
    
    # identify loose price
    loose_idx = findfirst(j -> a[j] != b[j], 1:length(a))
    x = zeros(D)
    for i in 1:D
        if i != loose_idx
            x[i] = a[i]
        end
    end

    # compute breakpoints the old way (with removed bounds condition)
    
    if !vectoro
        cp = Float64[]
        for idx in 1:(NS)
            # compute the utilities of all alternatives except loose_idx
            utils = [v_[idx,k] + theta_[idx,k] * x[k] for k in 1:D if k != loose_idx] 
            push!(utils, v_[idx,D+1])

            for l in 1:length(utils)
                bp = (utils[l] - v_[idx,loose_idx]) / theta_[idx,loose_idx]
                push!(cp, bp)
            end
        end
    else
        # Compute them in a vectorized way (Version 1)

        # Step 1: Compute the NSxD utils matrix
        utils_matrix = [v_[idx,k] + theta_[idx,k] * x[k] for idx=1:NS, k=1:D]

        # Step 2: Handle the D+1th utility
        # This creates a vector for the D+1th utility for each idx
        d_plus_one_utils = [v_[idx, D+1] for idx=1:NS]

        # Vectorized subtraction and division for the main matrix
        bp_matrix = (utils_matrix .- v_[:,loose_idx]) ./ theta_[:,loose_idx]

        # Flatten the matrix to a vector
        cp = vec(bp_matrix)

        # Append the D+1th utilities, adjusted similarly
        cp = [cp; (d_plus_one_utils .- v_[:,loose_idx]) ./ theta_[:,loose_idx]]    
    end

    ################################################
    
    # Filter all bp values outside the bounds
    cp = filter(x -> x >= a[loose_idx] && x <= b[loose_idx], cp)
	
	# dont forget to add bounds to breakpoints
	push!(cp, b[loose_idx])
	push!(cp, a[loose_idx])

    J = D

    # I) Initialize x_opt, Z_opt
    x_opt = copy(x) 
    Z_opt = 0

    # II) Test all the critical prices of loose_idx
    for xd in cp                          
        x[loose_idx] = xd

        if isnothing(prio_queue) 
            if !inv
                Z, _, _ = compute_obj_value_with_forced_capacities(x, zeros(D,D+1), phi_, theta_, caps)
            else
                Z, _, _ = compute_obj_value_with_forced_capacities(x, zeros(D,D+1), phi_, theta_, caps; inv)
            end
        else
            Z, _, _ = compute_obj_value_priority_queue(x, zeros(D,D+1), phi_, theta_, caps, prio_queue)
        end

        if(Z > Z_opt) # Update the incumbent solution
            Z_opt = Z
            x_opt = deepcopy(x)
        end
    end
    
    
    # Optimal solution
    return (x_opt, Z_opt/S)
end;

function do_poly_cap_one_price(exo_utility, endo_coef, S, prices, loose_index, a, b, caps, prio_queue; inv=false)
    D = size(exo_utility)[2] - 1
    a_ = copy(prices)
    b_ = copy(prices)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]
    (x_poly, Z_poly) = polytime_simple_multibreak_caps_simpler(exo_utility, endo_coef, S, a_, b_, caps, prio_queue; inv)
    return (x_poly[loose_index], Z_poly)
end;

function polytime_cap_heuristic(exo_utility, endo_coef, S, start_prices, a, b, caps, prio_queue=nothing; inv=false)
    # initialize prices
    D = size(exo_utility)[2] - 1
    start_prices = Float64.(start_prices) # cast prices to float in case the start prices were all integer
    prices = copy(start_prices)
    
    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1
    best_obj = 0
    
    best_prices = copy(prices)
    consecutive_close = 0
    old_obj = 0
    
    # Define a threshold for convergence
    epsilon = 1e-9
    iter = 0 # failsafe to not run forever in case of non-convergence
    
    obj_val = best_obj
    
    while consecutive_close < D && iter < 20
        iter += 1
        
        # Call do_poly_one_price to get x_poly and obj_val
        # loose_price, obj_val = do_poly_one_price(exo_utility, endo_coef, prices, loose_index, [a[loose_index]], [b[loose_index]])
        # loose_price, obj_val = do_poly_one_price_alt(exo_utility, endo_coef, prices, loose_index, [a[loose_index]], [b[loose_index]])

        loose_price, obj_val = do_poly_cap_one_price(exo_utility, endo_coef, S, prices, loose_index, [a[loose_index]], [b[loose_index]], caps, prio_queue; inv)
        
        # Update the price at the current index
        prices[loose_index] = loose_price

        if obj_val > best_obj
            best_obj = obj_val
            best_prices = copy(prices)
        end
        
        # Check if the current obj_val is very close to the previous one
        if abs(obj_val - old_obj) < epsilon
            consecutive_close += 1 # Increment the counter
        else
            consecutive_close = 0  # Reset consecutive_close counter
        end
        
        old_obj = obj_val
        
        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > D
            loose_index = 1
        end
    end
    
    return best_obj, best_prices, iter
end;

function line_search(exo_utility, endo_coef, a, b, curr_prices, curr_obj)
    N,S,C = size(exo_utility)
	NS = N*S
    D = C-1
    line_search_delta = 0.015
    steps = 3
    max_dev = line_search_delta * steps
    iterations = 0
    improvement = true
    best_prices = curr_prices
    best_obj = curr_obj
	
    # exo_utility_  = reshape(exo_utility, NS, C)
    # endo_coef_  = reshape(endo_coef, NS, C)
	
    while improvement
        iterations += 1
        improvement = false
        
        # a = curr_prices .- max_dev
        # b = curr_prices .+ max_dev

        for j in 1:D
            for direction in [-1, 1]
                for increment in 1:steps
                    # Create a new candidate solution
                    new_prices = curr_prices
                    new_prices[j] += direction * increment * line_search_delta
					
					if a[j] <= new_prices[j] <= b[j]
                    	# Evaluate the candidate solution
                    	new_obj, new_prices, iter = polytime_heuristic(exo_utility, endo_coef, new_prices, a, b)
                    
                    	# Update best solution if the new one is better
                    	if new_obj >= best_obj + 1e-9 && all(a[j] <= new_prices[j] <= b[j] for j in 1:D)
                        	best_obj = new_obj
                        	best_prices = new_prices
                        	improvement = true
						end
					end
                end
            end
        end
        # after performing the line search, update the current solution
        # if there was an improvement, we will automatically perform the next iteration, if not, we terminate
        curr_prices = best_prices
        curr_obj = best_obj
    end
    return best_obj, best_prices, iterations
end;

function extended_line_search(exo_utility, endo_coef, a, b, curr_prices, curr_obj)
    N,S,C = size(exo_utility)
    D = C-1
    line_search_delta = 0.015
    step_increase_factor = 2
    max_line_search_delta = 0.4  # Define a maximum step size
    steps = 3
    iterations = 0
    improvement = true
    best_prices = curr_prices
    best_obj = curr_obj
    no_improvement_streak = 0  # Keep track of consecutive non-improvement iterations
	
    # exo_utility_  = reshape(exo_utility, NS, C)
    # endo_coef_  = reshape(endo_coef, NS, C)

    while improvement || line_search_delta < max_line_search_delta
        iterations += 1
        improvement = false
        
        #a = curr_prices .- max_line_search_delta
        #b = curr_prices .+ max_line_search_delta

        for j in 1:length(curr_prices)
            for direction in [-1, 1]
                for increment in 1:steps
                    # Create a new candidate solution
                    new_prices = copy(curr_prices)
                    new_prices[j] += direction * increment * line_search_delta
					
					if a[j] <= new_prices[j] <= b[j]
                    	# Evaluate the candidate solution
                   	 	new_obj, new_prices, iter = polytime_heuristic(exo_utility, endo_coef, new_prices, a, b)
                    
                   		# Update best solution if the new one is better
                    	if new_obj > best_obj + 1e-9 && all(a[j] <= new_prices[j] <= b[j] for j in 1:D)
                        	best_obj = new_obj
                        	best_prices = new_prices
                        	improvement = true
                        	no_improvement_streak = 0  # Reset the no-improvement streak
                    	end
					end
                end
            end
        end

        # Check if there was an improvement in this iteration
        if !improvement
            no_improvement_streak += 1
            line_search_delta *= step_increase_factor  # Increase the step size
            println("No further improvement found. Increasing step size to $line_search_delta.")
        end

        # after performing the line search, update the current solution
        curr_prices = best_prices
        curr_obj = best_obj

        # Stopping condition based on a maximum number of consecutive no-improvement iterations or maximum step size
        if line_search_delta >= max_line_search_delta #  || no_improvement_streak >= 5
            println("Stopping heuristic after $iterations iterations with no_improvement_streak of $no_improvement_streak.")
            break
        end
    end
    return best_obj, best_prices, iterations
end;

function line_search_cap(exo_utility, endo_coef, S, a, b, curr_prices, curr_obj, caps, prio_queue=nothing; inv=false)
    NS,C = size(exo_utility)
    D = C-1
    line_search_delta = 0.005
    steps = 3
    max_dev = line_search_delta * steps
    iterations = 0
    improvement = true
    best_prices = curr_prices
    best_obj = curr_obj
	
    while improvement
        iterations += 1
        improvement = false
        
        #a = curr_prices .- max_dev
        #b = curr_prices .+ max_dev

        for j in 1:D
            for direction in [-1, 1]
                for increment in 1:steps
                    # Create a new candidate solution
                    new_prices = curr_prices
                    new_prices[j] += direction * increment * line_search_delta
					
					if a[j] <= new_prices[j] <= b[j]
                    	# Evaluate the candidate solution
                    	#println("evaluating start prices = $new_prices")
                    	new_obj, new_prices, iter = polytime_cap_heuristic(exo_utility, endo_coef, S, new_prices, a, b, caps, prio_queue; inv)
                    	#println("gave obj = $new_obj")
                    	# Update best solution if the new one is better
                    	if new_obj >= best_obj + 1e-9 && all(a[j] <= new_prices[j] <= b[j] for j in 1:D)
                        	best_obj = new_obj
                        	best_prices = new_prices
                        	improvement = true
                    	end
					end
                end
            end
        end
        # after performing the line search, update the current solution
        # if there was an improvement, we will automatically perform the next iteration, if not, we terminate
        curr_prices = best_prices
        curr_obj = best_obj
    end
    return best_obj, best_prices, iterations
end;

function extended_line_search_cap(exo_utility, endo_coef, S, a, b, curr_prices, curr_obj, caps, prio_queue=nothing; inv=false)
    NS,C = size(exo_utility)
    D = C-1
    line_search_delta = 0.005
    step_increase_factor = 2
    max_line_search_delta = 0.05  # Define a maximum step size
    steps = 3
    iterations = 0
    improvement = true
    best_prices = curr_prices
    best_obj = curr_obj
    no_improvement_streak = 0  # Keep track of consecutive non-improvement iterations

    while improvement || line_search_delta < max_line_search_delta
        iterations += 1
        improvement = false
        
        #a = curr_prices .- max_line_search_delta
        #b = curr_prices .+ max_line_search_delta

        for j in 1:length(curr_prices)
            for direction in [-1, 1]
                for increment in 1:steps
                    # Create a new candidate solution
                    new_prices = copy(curr_prices)
                    new_prices[j] += direction * increment * line_search_delta
					
					if a[j] <= new_prices[j] <= b[j]
                    	# Evaluate the candidate solution
                    	new_obj, new_prices, iter = polytime_cap_heuristic(exo_utility, endo_coef, S, new_prices, a, b, caps, prio_queue; inv)
                    
                    	# Update best solution if the new one is better
                    	if new_obj > best_obj + 1e-9 && all(a[j] <= new_prices[j] <= b[j] for j in 1:D)
                        	best_obj = new_obj
                        	best_prices = new_prices
                        	improvement = true
                        	no_improvement_streak = 0  # Reset the no-improvement streak
                    	end
					end
                end
            end
        end

        # Check if there was an improvement in this iteration
        if !improvement
            no_improvement_streak += 1
            line_search_delta *= step_increase_factor  # Increase the step size
            println("No further improvement found. Increasing step size to $line_search_delta.")
        else
            println("Found an improvement")
            no_improvement_streak = 0
        end

        # after performing the line search, update the current solution
        curr_prices = best_prices
        curr_obj = best_obj

        # Stopping condition based on a maximum number of consecutive no-improvement iterations or maximum step size
        if line_search_delta >= max_line_search_delta #  || no_improvement_streak >= 5
            println("Stopping heuristic after $iterations iterations with no_improvement_streak of $no_improvement_streak.")
            break
        end
    end
    return best_obj, best_prices, iterations
end;

function run_algo(N, S, J_PSP, J_PUP)
    N_orig = copy(N)
    S_orig = copy(S)
    
    # init
	data = npzread("init_data/data_test_5_5_1_1.npz")["arr"]
    exo_utility = data[1, :, :, :];
    endo_coef = data[2, :, :, :];
    exo_utility = permutedims(exo_utility, [3, 2, 1]);
    endo_coef = permutedims(endo_coef, [3, 2, 1]);

    N = size(exo_utility)[1]
    S = size(exo_utility)[2]
    D = size(exo_utility)[3] - 1
	C = D+1

    phi = exo_utility; # exo_util, has shape NRJ
    theta = endo_coef; # endo_coefficient, has shape NRJ, last entry 0 / 1

    h = zeros(N,D+1); # reset so that u[n,s,c] = h[n,c]+phi[n,s,c]+theta[n,s,c]*x[c] is what it should be
	
	a = [0.5, 0.65]
	b = [0.7, 0.85]
	
	caps = [2 * S for _ in 1:D]
	
    prio_queue = 1:(N*S)
	
    println("Starting initialization, N=$N, S=$S")
    println("")
    println("")
    
    start_prices = (a + b) / 2
    time_poly = (@timed begin (x_poly, Z_poly) = polytime_simple_multibreak_caps(h, phi, theta, a, b, caps, prio_queue)  end).time
    println("time_poly              = ",time_poly)
    println("x_poly                 = ",x_poly)
    println("Z_poly                 = ",Z_poly)
    
    println("")
    println("")
    println("Finished initialization")
    println("")
    N = N_orig
    S = S_orig
	
	
	# # Define the base values
	base_a = [0.5, 0.65]
	base_b = [0.7, 0.85]

	a = [repeat([base_a[i]], J_PSP + (i-1) * (J_PUP - J_PSP)) for i in 1:length(base_a)]
	b = [repeat([base_b[i]], J_PSP + (i-1) * (J_PUP - J_PSP)) for i in 1:length(base_b)]

	# Flatten the arrays to form a single level array
	a = vcat(a...)
	b = vcat(b...)
	
    prio_queue = 1:(N*S)
	
    println("Running real instance, N=$N, S=$S")
    println("")
    println("")

    data = npzread("data_test_$(N)_$(S)_$(J_PSP)_$(J_PUP).npz")["arr"]
	
    exo_utility = data[1, :, :, :];
    endo_coef = data[2, :, :, :];
    
    exo_utility = permutedims(exo_utility, [3, 2, 1]);
    endo_coef = permutedims(endo_coef, [3, 2, 1]);

    N = size(exo_utility)[1]
    S = size(exo_utility)[2]
    D = size(exo_utility)[3] - 1
	
	if N == 50
		caps = [20 * S for _ in 1:D]
	elseif N == 100
		caps = [40 * S for _ in 1:D]
	elseif N == 150
		caps = [60 * S for _ in 1:D]
	elseif N == 198
		caps = [80 * S for _ in 1:D]
	else
		caps = [floor(N * 0.4) * S for _ in 1:D]
	end

    phi = exo_utility; # exo_util, has shape NRJ
    theta = endo_coef; # endo_coefficient, has shape NRJ, last entry 0 / 1

    h = zeros(N,D+1); # reset so that u[n,s,c] = h[n,c]+phi[n,s,c]+theta[n,s,c]*x[c] is what it should be
   
    time_poly = (@timed begin (x_poly, Z_poly) = polytime_simple_multibreak_caps(h, phi, theta, a, b, caps, prio_queue)  end).time
    println("time_poly              = ",time_poly)
    println("x_poly                 = ",x_poly)
    println("Z_poly                 = ",Z_poly)
		
end;
vectoro = false;

N = parse(Int, ARGS[1])
S = parse(Int, ARGS[2])
J_PSP = parse(Int, ARGS[3])
J_PUP = parse(Int, ARGS[4])

run_algo(N, S, J_PSP, J_PUP)
