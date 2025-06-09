using Pkg

Pkg.activate(".")

using Evolutionary
using Combinatorics
using DataStructures
# using DataFrames, CSV
using JSON3
using Random
using Distributions
using NPZ
using DelimitedFiles
using LinearAlgebra # used for bLL mixed logit computation apparently
using Statistics # for the mean? da f

function compute_sLL_mixed(x, y, av, epsilon, R, sigma, beta, mix_inds)
    NS = size(x, 2)
    N = Int(NS / R)
    K = size(x, 3)
    logOfZero = -100
    hits_idx = zeros(NS)
    hits = zeros(N)
    
    toll = 1e-6

    for idx in 1:NS
        y_idx = y[:, idx]
        obs_idx = findfirst(x -> x == 1, y_idx)    
        n = div(idx - 1, R) + 1
        av_idx = findall(av[:, idx] .== 1)

        max_u = -Inf
        max_j = -1
        for j in av_idx
            u = sum(x[j, idx, k] * beta[k] for k in 1:K) + sum(x[j, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[j, idx]

            # Track the maximum value and corresponding index
            if u > max_u + toll
                max_u = u
                max_j = j
            elseif abs(u - max_u) < toll && j == obs_idx
                max_u = u
                max_j = j
            end
        end

        if max_j == obs_idx
            hits_idx[idx] = 1
            hits[n] += 1
        end
    end
    
    return hits, hits_idx, sum((s = hits[n]; s > 0 ? log(s) : logOfZero) - log(R) for n in 1:N)
end;

function compute_sLL_latent3classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks, class_3_ks)
    J = size(x, 1)
    NS = size(x, 2)
    N = Int(NS / R)
    K = size(x, 3)
    C = 3
    logOfZero = -100
    hits_idx = zeros(NS)
    hits = zeros(N)
    
    latent_beta = beta[prob_inds[1]:prob_inds[2]]
    trueL = length(beta) - 2
    beta = beta[1:trueL]
    
    toll = 1e-6

    for idx in 1:NS
        y_idx = y[:, idx]
        obs_idx = findfirst(x -> x == 1, y_idx)    
        n = div(idx - 1, R) + 1
        av_idx = findall(av[:, idx] .== 1) 

        if class_draws[idx] <= latent_beta[1]
            c = 1
        elseif class_draws[idx] <= latent_beta[2]
            c = 2
        else
            c = 3
        end

        max_u = -Inf
        max_j = -1
        for j in av_idx
            if c == 1
                u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
            elseif c == 2
                u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
            else
                u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
            end

            # Track the maximum value and corresponding index
            if u > max_u + toll
                max_u = u
                max_j = j
            elseif abs(u - max_u) < toll && j == obs_idx
                max_u = u
                max_j = j
            end
        end

        if max_j == obs_idx
            hits_idx[idx] = 1
            hits[n] += 1
        end
    end
    
    return hits, hits_idx, sum((s = hits[n]; s > 0 ? log(s) : logOfZero) - log(R) for n in 1:N)
end;

function compute_sLL_latent2classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks)
    J = size(x, 1)
    NS = size(x, 2)
    N = Int(NS / R)
    K = size(x, 3)
    C = 2
    logOfZero = -100
    hits_idx = zeros(NS)
    hits = zeros(N)
    
    latent_beta = beta[prob_inds[1]]
    trueL = length(beta) - 1
    beta = beta[1:trueL]
    
    toll = 1e-6

    for idx in 1:NS
        y_idx = y[:, idx]
        obs_idx = findfirst(x -> x == 1, y_idx)    
        n = div(idx - 1, R) + 1
        av_idx = findall(av[:, idx] .== 1)
        
        if class_draws[idx] <= latent_beta[1]
            c = 1
        else
            c = 2
        end

        max_u = -Inf
        max_j = -1
        for j in av_idx
            if c == 1
                u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
            else
                u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
            end

            # Track the maximum value and corresponding index
            if u > max_u + toll
                max_u = u
                max_j = j
            elseif abs(u - max_u) < toll && j == obs_idx
                max_u = u
                max_j = j
            end
        end

        if max_j == obs_idx
            hits_idx[idx] = 1
            hits[n] += 1
        end
    end
    
    return hits, hits_idx, sum((s = hits[n]; s > 0 ? log(s) : logOfZero) - log(R) for n in 1:N)
end;

function compute_sLL_latent4classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks, class_3_ks, class_4_ks)
    J = size(x, 1)
    NS = size(x, 2)
    N = Int(NS / R)
    K = size(x, 3)
    C = 4
    logOfZero = -100
    hits_idx = zeros(NS)
    hits = zeros(N)
    
    latent_beta = beta[prob_inds[1]:prob_inds[3]]
    trueL = length(beta) - 3
    beta = beta[1:trueL]
    
    toll = 1e-6

    for idx in 1:NS
        y_idx = y[:, idx]
        obs_idx = findfirst(x -> x == 1, y_idx)    
        n = div(idx - 1, R) + 1
        av_idx = findall(av[:, idx] .== 1)

        if class_draws[idx] <= latent_beta[1]
            c = 1
        elseif class_draws[idx] <= latent_beta[2]
            c = 2
        elseif class_draws[idx] <= latent_beta[3]
            c = 3
        else
            c = 4
        end

        max_u = -Inf
        max_j = -1
        for j in av_idx
            if c == 1
                u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
            elseif c == 2
                u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
            elseif c == 3
                u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
            else
                u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
            end

            # Track the maximum value and corresponding index
            if u > max_u + toll
                max_u = u
                max_j = j
            elseif abs(u - max_u) < toll && j == obs_idx
                max_u = u
                max_j = j
            end
        end

        if max_j == obs_idx
            hits_idx[idx] = 1
            hits[n] += 1
        end
    end
    
    return hits, hits_idx, sum((s = hits[n]; s > 0 ? log(s) : logOfZero) - log(R) for n in 1:N)
end;

function compute_sLL_latent5classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks)
    J = size(x, 1)
    NS = size(x, 2)
    N = Int(NS / R)
    K = size(x, 3)
    C = 5
    logOfZero = -100
    hits_idx = zeros(NS)
    hits = zeros(N)
    
    latent_beta = beta[prob_inds[1]:prob_inds[4]]
    trueL = length(beta) - 4
    beta = beta[1:trueL]
    
    toll = 1e-6

    for idx in 1:NS
        y_idx = y[:, idx]
        obs_idx = findfirst(x -> x == 1, y_idx)    
        n = div(idx - 1, R) + 1
        av_idx = findall(av[:, idx] .== 1)

        if class_draws[idx] <= latent_beta[1]
            c = 1
        elseif class_draws[idx] <= latent_beta[2]
            c = 2
        elseif class_draws[idx] <= latent_beta[3]
            c = 3
        elseif class_draws[idx] <= latent_beta[4]
            c = 4
        else
            c = 5
        end

        max_u = -Inf
        max_j = -1
        for j in av_idx
            if c == 1
                u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
            elseif c == 2
                u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
            elseif c == 3
                u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
            elseif c == 4
                u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
            else
                u = sum(x[j, idx, k] * beta[k] for k in class_5_ks) + epsilon[j, idx]
            end

            # Track the maximum value and corresponding index
            if u > max_u + toll
                max_u = u
                max_j = j
            elseif abs(u - max_u) < toll && j == obs_idx
                max_u = u
                max_j = j
            end
        end

        if max_j == obs_idx
            hits_idx[idx] = 1
            hits[n] += 1
        end
    end
    
    return hits, hits_idx, sum((s = hits[n]; s > 0 ? log(s) : logOfZero) - log(R) for n in 1:N)
end;

function do_BHAMSLE_onedim_mixed(x, y, av, epsilon, sigma, R, beta, loose_index, mix_inds, a, b, bp_proximity)    
    # for now we decided that reducing to one dim eh I mean its not a helpful middle step
    a_ = copy(beta)
    b_ = copy(beta)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]

    (beta_poly, sLL_poly) = BHAMSLE_onedim_mixed(x, y, av, epsilon, sigma, R, beta, mix_inds, a_, b_)
    return (beta_poly[loose_index], sLL_poly)
end;

function do_BHAMSLE_onedim_latent3classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, class_3_ks, prob_inds, a, b, bp_proximity, class_1_av, class_2_av, class_3_av)    
    # for now we decided that reducing to one dim eh I mean its not a helpful middle step
    a_ = copy(beta)
    b_ = copy(beta)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]
    
    (beta_poly, sLL_poly) = BHAMSLE_onedim_latent3classes(x, y, av, epsilon, class_draws, R, beta, class_1_ks, class_2_ks, class_3_ks, prob_inds, a_, b_, class_1_av, class_2_av, class_3_av)
    return (beta_poly[loose_index], sLL_poly)
end;

function do_BHAMSLE_onedim_latent2classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, prob_inds, a, b, bp_proximity, class_1_av, class_2_av)    
    # for now we decided that reducing to one dim eh I mean its not a helpful middle step
    a_ = copy(beta)
    b_ = copy(beta)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]

    (beta_poly, sLL_poly) = BHAMSLE_onedim_latent2classes(x, y, av, epsilon, class_draws, R, beta, class_1_ks, class_2_ks, prob_inds, a_, b_, class_1_av, class_2_av)
    return (beta_poly[loose_index], sLL_poly)
end;

function do_BHAMSLE_onedim_latent4classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, class_3_ks, class_4_ks, prob_inds, a, b, bp_proximity)    
    # for now we decided that reducing to one dim eh I mean its not a helpful middle step
    a_ = copy(beta)
    b_ = copy(beta)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]

    (beta_poly, sLL_poly) = BHAMSLE_onedim_latent4classes(x, y, av, epsilon, class_draws, R, beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, prob_inds, a_, b_)
    return (beta_poly[loose_index], sLL_poly)
end;

function do_BHAMSLE_onedim_latent5classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, prob_inds, a, b, bp_proximity)    
    # for now we decided that reducing to one dim eh I mean its not a helpful middle step
    a_ = copy(beta)
    b_ = copy(beta)
    a_[loose_index] = a[1]
    b_[loose_index] = b[1]

    (beta_poly, sLL_poly) = BHAMSLE_onedim_latent5classes(x, y, av, epsilon, class_draws, R, beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, prob_inds, a_, b_)
    return (beta_poly[loose_index], sLL_poly)
end;

function BHAMSLE_onedim_latent3classes(x, y, av, epsilon, sigma, R, beta, class_1_ks, class_2_ks, class_3_ks, prob_inds, a, b, class_1_av, class_2_av, class_3_av)    
    # Problem size
    NS = size(epsilon)[2]
    J = size(x)[1]
    K = size(x)[3]
    N = Int(NS / R)

    #Initial solution
    beta = zeros(length(beta))
    logOfZero = -100
	iteration = 0
    
    # identify loose price AND define beta I guess
    loose_idx = findfirst(j -> a[j] != b[j], 1:length(a))
    
    for i in 1:length(a)
        if i != loose_idx
            beta[i] = a[i]
        end
    end
    
    # BoundsError: attempt to access 1-element Vector{Float64} at index [2]
    
    # compute breakpoints 
    cp = []
    
    # initialize the greatest hits
    hits = zeros(N)
    
    # time_build_cp = (@timed begin
    
    toll = 1e-6
    
    plusslack = -0.999e-6  # (tol - epsilon) this should be 0.999* the utility comparison tolerance
    # we changed its sign too so that we can just do the same stuff as we did for sunlight
    # without getting confused
    
    sunlight = -1.5e-6 # (tol + epsilon) set to 0 to activate shadow
    # funny story, we subtract the sunlight from obs, soo at the bp we get obs = compet + sunlight
    # so if we have sunlight positive thats wrong, the point is that obs is at least sunlight WORSE
    # which is why we have to make it negative
    
    # this determines the amount of sunlight. Should be ever so slightly more than the
    # sLL computation threshold of comparing utilties. Controls when a minusser is removed
    # (when the observed utility is "sunlight" away from best competition)
    
    if !(loose_idx in prob_inds) 
        #     if sigma < beta[7]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
        #     elseif sigma < beta[8]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
        #     else
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
        #     end
        # end
        
        for idx in 1:(NS)   
            av_idx = findall(av[:, idx] .== 1) # new
#             if !(sigma[idx] ≈ beta[7]) && !(sigma[idx] ≈ beta[8])
            # actually I decided that we wont bother with this nitpicking. Its a heuristic.
            if sigma[idx] <= beta[prob_inds[1]] # class 1
                av_idx = intersect(class_1_av, av_idx)
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_1_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0 && x in av_idx, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx # new
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx # new
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                           if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                    
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                    # we had to do the bounds check first
                    
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end      
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx # new
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                             if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            elseif sigma[idx] <= beta[prob_inds[2]] # class 2
                av_idx = intersect(class_2_av, av_idx)
       # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_2_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx # new
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0 && x in av_idx, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx # new
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx # new
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                        
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end


                    # 3.) compute breakpoint: 
                            
                    # lets think about. it if loose idx == 6, EVERY alt is influenced, since its time.
                    # then eeh yeah I mean you should divide by stressfactor too wtf. Then we dont need it in the constant LHS
                            

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                            
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end              
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            else # class 3
                av_idx = intersect(class_3_av, av_idx)
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_3_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0 && x in av_idx, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                            
                    # is the individual a plusser or a minusser?
                    plusser = true  
					
					if class_3_ks == [loose_idx]
						c_total = 0
					else
						c_total = sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if k != loose_idx)
					end
											                  
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            breakpoint = ((max_u - sunlight) - (c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end
                                
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end    
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_3_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end
                    
                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            end
        end
    else  # estimating latent class probabilites
        # compute the utility everyone gets with the different classes (and which one is the chosen in each case)
        for idx in 1:(NS)   
            av_idx = findall(av[:, idx] .== 1)
            
            n = div(idx - 1, R) + 1
            winners_classes = []

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)
            
            for c in 1:3
                max_u = -Inf
                max_j = -1
                for j in av_idx
                    if c == 1
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                    elseif c == 2
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                    else
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                    end
                    # Track the maximum value and corresponding index
                    if u > max_u + toll
                        max_u = u
                        max_j = j
                    elseif abs(u - max_u) < toll && j == obs_idx
                        max_u = u
                        max_j = j
                    end
                end

                if max_j == obs_idx
                    push!(winners_classes, c)
                end
            end
            
            if winners_classes == [1, 2, 3]
                hits[n] += 1
                # no matter the breakpoint we always get this gentleman
                continue
            end

            if winners_classes == []
                # no matter the breakpoint we always loose this notgentleman
                continue
            end

            if loose_idx == prob_inds[1] # this influences the separator from class 1 to 2 
                if sigma[idx] > beta[prob_inds[2]]
                    if 3 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else
                    if (1 in winners_classes) && !(2 in winners_classes) # only class 1 wins
                        # add breakpoint = sigma[idx] as a plusser, since as soon as beta[7] > sigma[idx], we put him in
                        # class 1 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (2 in winners_classes) && !(1 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[7] > sigma[idx], we put him in
                        # class 1 and thus loose him
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (1 in winners_classes) && (2 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(1 in winners_classes) && !(2 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            else # this influences the separator from class 2 to 3 
                if sigma[idx] <= beta[prob_inds[1]]
                    if 1 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else   
                    if (2 in winners_classes) && !(3 in winners_classes) # only class 2 wins
                        # add breakpoint = sigma[idx] as a plusser since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (3 in winners_classes) && !(2 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and we loose him 
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (2 in winners_classes) && (3 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(2 in winners_classes) && !(3 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            end
        end
    end

    # end).time
    
    # println("building cp took $time_build_cp s")
    
    sLL_tot = - N * log(R) + sum((s = hits[n]; s > 0 ? log(s) : -100) for n in 1:N)    
    sLL_opt = - N * log(R) # this makes more sense no?
    beta_opt = copy(beta)
    
    # II) Test all the critical value of loose_idx
    
    # sorting
    sorted_cp = sort(cp, by = x -> x[1])  
    
    for xd in sorted_cp
        n = Int(xd[2])
        idx = Int(xd[3])
        m = hits[n]
        beta[loose_idx] = xd[1]

        if Bool(xd[4]) # if plusser
            if m == 0
                sLL_tot += log(m+1) - (-100)
                hits[n] += 1
            else
                sLL_tot += log(m+1) - log(m)
                hits[n] += 1 
            end
        else # if minusser
            # sunlight means immediate minuss procedure
            if m == 1 
                sLL_tot += (-100) - log(m)
                hits[n] -= 1
            else
                sLL_tot += log(m-1) - log(m)
                hits[n] -= 1
            end
        end

        if sLL_tot > sLL_opt
           sLL_opt = sLL_tot
           beta_opt = deepcopy(beta)
        end
    end
            
    # Optimal solution
    return beta_opt, sLL_opt
end;

function BHAMSLE_onedim_latent2classes(x, y, av, epsilon, sigma, R, beta, class_1_ks, class_2_ks, prob_inds, a, b, class_1_av, class_2_av)    
    # Problem size
    NS = size(epsilon)[2]
    J = size(x)[1]
    K = size(x)[3]
    N = Int(NS / R)

    #Initial solution
    beta = zeros(length(beta))
    logOfZero = -100
	iteration = 0
    
    # identify loose price AND define beta I guess
    loose_idx = findfirst(j -> a[j] != b[j], 1:length(a))
    
    for i in 1:length(a)
        if i != loose_idx
            beta[i] = a[i]
        end
    end
    
    # BoundsError: attempt to access 1-element Vector{Float64} at index [2]
    
    # compute breakpoints 
    cp = []
    
    # initialize the greatest hits
    hits = zeros(N)
    
    # time_build_cp = (@timed begin
    
    toll = 1e-6
    
    plusslack = -0.999e-6  # (tol - epsilon) this should be 0.999* the utility comparison tolerance
    # we changed its sign too so that we can just do the same stuff as we did for sunlight
    # without getting confused
    
    sunlight = -1.5e-6 # (tol + epsilon) set to 0 to activate shadow
    # funny story, we subtract the sunlight from obs, soo at the bp we get obs = compet + sunlight
    # so if we have sunlight positive thats wrong, the point is that obs is at least sunlight WORSE
    # which is why we have to make it negative
    
    # this determines the amount of sunlight. Should be ever so slightly more than the
    # sLL computation threshold of comparing utilties. Controls when a minusser is removed
    # (when the observed utility is "sunlight" away from best competition)
    
    if !(loose_idx in prob_inds) 
        #     if sigma < beta[7]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
        #     elseif sigma < beta[8]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
        #     else
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
        #     end
        # end
        
        for idx in 1:(NS)   
            # set of available alternatives for this guy
            # we should easily be able to manipulate that 
            # lets say 
            # class_1_av = [1, 2, 3]
            # class_2_av = [1, 2]
            # class_3_av = [1, 2, 3]
            #
            # av_idx = intersection between class_av and av_idx
            av_idx = findall(av[:, idx] .== 1)
#             if !(sigma[idx] ≈ beta[7]) && !(sigma[idx] ≈ beta[8])
            # actually I decided that we wont bother with this nitpicking. Its a heuristic.
            if sigma[idx] <= beta[prob_inds[1]] # class 1
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                av_idx = intersect(class_1_av, av_idx)
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_1_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                           if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                    
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                    # we had to do the bounds check first
                    
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end      
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                             if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            else # class 2
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
                av_idx = intersect(class_2_av, av_idx)
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_2_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                            
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    				
					if class_2_ks == [loose_idx]
						c_total = 0
					else
						c_total = sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if k != loose_idx)
					end
						
					# you're a minusser if:	
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            breakpoint = ((max_u - sunlight) - (c_total + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end
					

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end
                                
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end    
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_2_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end
                    
                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            end
        end
    else  # estimating latent class probabilites
        # compute the utility everyone gets with the different classes (and which one is the chosen in each case)
        for idx in 1:(NS)  
            av_idx = findall(av[:, idx] .== 1) 
            
            n = div(idx - 1, R) + 1
            winners_classes = []

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)
            
            for c in 1:2
                max_u = -Inf
                max_j = -1
                for j in av_idx
                    if c == 1
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                    else
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                    end
                    # Track the maximum value and corresponding index
                    if u > max_u + toll
                        max_u = u
                        max_j = j
                    elseif abs(u - max_u) < toll && j == obs_idx
                        max_u = u
                        max_j = j
                    end
                end

                if max_j == obs_idx
                    push!(winners_classes, c)
                end
            end
            
            if winners_classes == [1, 2]
                hits[n] += 1
                # no matter the breakpoint we always get this gentleman
                continue
            end

            if winners_classes == []
                # no matter the breakpoint we always loose this notgentleman
                continue
            end


            if (1 in winners_classes) && !(2 in winners_classes) # only class 1 wins
                # add breakpoint = sigma[idx] as a plusser, since as soon as beta[7] > sigma[idx], we put him in
                # class 1 and thus we win
                plusser = true
                push!(cp, [sigma[idx], n, idx, plusser])
            elseif (2 in winners_classes) && !(1 in winners_classes)
                # add breakpoint = sigma[idx] as a minusser, since as soon as beta[7] > sigma[idx], we put him in
                # class 1 and thus loose him
                hits[n] += 1
                plusser = false
                push!(cp, [sigma[idx], n, idx, plusser])
            elseif (1 in winners_classes) && (2 in winners_classes)
                hits[n] += 1
                # no matter the breakpoint we always get this gentleman
                continue
            elseif !(1 in winners_classes) && !(2 in winners_classes)
                continue # no matter what we always loose
            end
        end
    end

    # end).time
    
    # println("building cp took $time_build_cp s")
    
    sLL_tot = - N * log(R) + sum((s = hits[n]; s > 0 ? log(s) : -100) for n in 1:N)    
    sLL_opt = - N * log(R) # this makes more sense no?
    beta_opt = copy(beta)
    
    # II) Test all the critical value of loose_idx
    
    # sorting
    sorted_cp = sort(cp, by = x -> x[1])  
	
    for xd in sorted_cp
        n = Int(xd[2])
        idx = Int(xd[3])
        m = hits[n]
        beta[loose_idx] = xd[1]

        if Bool(xd[4]) # if plusser
            if m == 0
                sLL_tot += log(m+1) - (-100)
                hits[n] += 1
            else
                sLL_tot += log(m+1) - log(m)
                hits[n] += 1 
            end
        else # if minusser
            # sunlight means immediate minuss procedure
            if m == 1 
                sLL_tot += (-100) - log(m)
                hits[n] -= 1
            else
                sLL_tot += log(m-1) - log(m)
                hits[n] -= 1
            end
        end

        if sLL_tot > sLL_opt
           sLL_opt = sLL_tot
           beta_opt = deepcopy(beta)
        end
    end
            
    # Optimal solution
    return beta_opt, sLL_opt
end;

function BHAMSLE_onedim_latent4classes(x, y, av, epsilon, sigma, R, beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, prob_inds, a, b)    
    # Problem size
    NS = size(epsilon)[2]
    J = size(x)[1]
    K = size(x)[3]
    N = Int(NS / R)

    #Initial solution
    beta = zeros(length(beta))
    logOfZero = -100
	iteration = 0
    
    # identify loose price AND define beta I guess
    loose_idx = findfirst(j -> a[j] != b[j], 1:length(a))
    
    for i in 1:length(a)
        if i != loose_idx
            beta[i] = a[i]
        end
    end
    
    # BoundsError: attempt to access 1-element Vector{Float64} at index [2]
    
    # compute breakpoints 
    cp = []
    
    # initialize the greatest hits
    hits = zeros(N)
    
    # time_build_cp = (@timed begin
    
    toll = 1e-6
    
    plusslack = -0.999e-6  # (tol - epsilon) this should be 0.999* the utility comparison tolerance
    # we changed its sign too so that we can just do the same stuff as we did for sunlight
    # without getting confused
    
    sunlight = -1.5e-6 # (tol + epsilon) set to 0 to activate shadow
    # funny story, we subtract the sunlight from obs, soo at the bp we get obs = compet + sunlight
    # so if we have sunlight positive thats wrong, the point is that obs is at least sunlight WORSE
    # which is why we have to make it negative
    
    # this determines the amount of sunlight. Should be ever so slightly more than the
    # sLL computation threshold of comparing utilties. Controls when a minusser is removed
    # (when the observed utility is "sunlight" away from best competition)
    
    if !(loose_idx in prob_inds) 
        #     if sigma < beta[7]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
        #     elseif sigma < beta[8]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
        #     else
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
        #     end
        # end
        
        for idx in 1:(NS)   
            av_idx = findall(av[:, idx] .== 1) 
#             if !(sigma[idx] ≈ beta[7]) && !(sigma[idx] ≈ beta[8])
            # actually I decided that we wont bother with this nitpicking. Its a heuristic.
            if sigma[idx] <= beta[prob_inds[1]] # class 1
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_1_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                           if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                    
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                    # we had to do the bounds check first
                    
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end      
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                             if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            elseif sigma[idx] <= beta[prob_inds[2]] # class 2
       # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_2_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                        
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end


                    # 3.) compute breakpoint: 
                            
                    # lets think about. it if loose idx == 6, EVERY alt is influenced, since its time.
                    # then eeh yeah I mean you should divide by stressfactor too wtf. Then we dont need it in the constant LHS
                            

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                            
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end              
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            elseif sigma[idx] <= beta[prob_inds[3]] # class 3
                # #     do all the checking with influenced alts and whatnot with this utility:
                 #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
                         n = div(idx - 1, R) + 1
                         
                         if !(loose_idx in class_3_ks) # class 1 is unaffected by beta_RH
                             y_idx = y[:, idx]
                             obs_idx = findfirst(x -> x == 1, y_idx)
         
                             # 1.) compute index and utility of best constant alternative     
                             max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                             max_j = -1     # Initialize index of max value
         
                             # compute utilities, which are ALL constant
                             for j in av_idx
                                 u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                                 # Track the maximum value and corresponding index
                                if u > max_u + toll
                                     max_u = u
                                     max_j = j
                                 elseif abs(u - max_u) < toll && j == obs_idx
                                     max_u = u
                                     max_j = j
                                 end
                             end
         
                             if obs_idx == max_j
                                 hits[n] += 1
                                 continue
                             else
                                 continue
                             end
                         end
                         
                         x_idx = x[:, idx, loose_idx]
                         influenced_alts = findall(x -> x != 0, x_idx)
         
                         if isempty(influenced_alts)
                             # if nothing is influenced by that loose thing, again it could be a case where the current idx
                             # for example is not part of that latent class. He still has utilities
         
                             y_idx = y[:, idx]
                             obs_idx = findfirst(x -> x == 1, y_idx)
         
                             # 1.) compute index and utility of best constant alternative     
                             max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                             max_j = -1     # Initialize index of max value
         
                             # compute utilities, which are ALL constant
                             for j in av_idx
                                 u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                                 # Track the maximum value and corresponding index
                                 if u > max_u + toll
                                     max_u = u
                                     max_j = j
                                 elseif abs(u - max_u) < toll && j == obs_idx
                                     max_u = u
                                     max_j = j
                                 end
                             end
                             # max_u and max_j are now what we were looking for
         
                             # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                             #     there is no way that we get LL contribution from this s, so skip it
         
                             if obs_idx == max_j
                                 hits[n] += 1
                                 continue
                             else
                                 continue
                             end
                         elseif length(influenced_alts) == 1
                             # so one influenced utility against the best constant one
         
                             # 1.) compute index and utility of best constant alternative   
         
                             influenced_alt = influenced_alts[1]
         
                             # max_j is the index of the constant alternative, and max_u is its utility value
         
                             y_idx = y[:, idx]
                             obs_idx = findfirst(x -> x == 1, y_idx)
         
                             max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                             max_j = -1     # Initialize index of max value
         
                             # compute all constant utilities
                             for j in av_idx
                                 if j != influenced_alt
                                     u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                                     # Track the maximum value and corresponding index
                                     if u > max_u + toll
                                         max_u = u
                                         max_j = j
                                     elseif abs(u - max_u) < toll && j == obs_idx
                                         max_u = u
                                         max_j = j
                                     end
                                 end
                             end
         
                            # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                             #     there is no way that we get LL contribution from this s, so skip it
         
                             if (obs_idx != max_j) && (obs_idx != influenced_alt)
                                 continue
                             end
                                 
                             # is the individual a plusser or a minusser?
                             plusser = true                    
                             # you're a minusser if:
                             if x[influenced_alt, idx, loose_idx] < 0 
                                 if obs_idx == influenced_alt 
                                     plusser = false
                                     breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                                 else                        
                                     breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                                 end
                             else
                                 if obs_idx == influenced_alt
                                     breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                                 else
                                     plusser = false
                                     # then max_u has to be the observed
                                     # either influenced or max_u is best, else we would have gotten stopped
                                     # at the condition right before
                                     breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                                 end
                             end
         
         
                             # 3.) compute breakpoint: 
                                     
                             # lets think about. it if loose idx == 6, EVERY alt is influenced, since its time.
                             # then eeh yeah I mean you should divide by stressfactor too wtf. Then we dont need it in the constant LHS
                                     
         
                             # we have to check if that breakpoint is within the bounds
         
                             if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                                 # then we should either count it as a hit or not, but in both cases skip this idx
                                 # we can choose any point in a,b to evaluate whether its a safe one or not
                                 # and again it depends on whether or not the observed is the constant or not
         
                                 infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx]
         
                                 if obs_idx == influenced_alt # observed is not constant
                                     if max_u <= infl_u
                                         # so the dependent is at least as good, we count it as a win
                                         hits[n] += 1
                                         continue
                                     else
                                         continue
                                     end
                                 else # observed is the best constant
                                     if max_u >= infl_u
                                         # so the constant is at least as good, we count it as a win
                                         hits[n] += 1
                                         continue
                                     else
                                         continue
                                     end
                                 end
                             end
         
                             # ok so here we now that the dude has a bp that is within the bounds
                                     
                             if plusser
                                 push!(cp, [breakpoint, n, idx, plusser])
                             else
                                 hits[n] += 1
                                 push!(cp, [breakpoint, n, idx, plusser])
                             end              
                         else    
                             # now we are in the interesting case, having multiple influenced alts
         
                             # 1.) compute index and utility of best constant alternative   
         
                             # max_j is the index of the constant alternative, and max_u is its utility value
         
                             y_idx = y[:, idx]
                             obs_idx = findfirst(x -> x == 1, y_idx)
         
                             max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                             max_j = -1     # Initialize index of max value
         
                             # compute all constant utilities
                             for j in av_idx
                                 if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                                     u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                                     # Track the maximum value and corresponding index
                                     if u > max_u + toll
                                         max_u = u
                                         max_j = j
                                     elseif abs(u - max_u) < toll && j == obs_idx
                                         max_u = u
                                         max_j = j
                                     end
                                 end
                             end
         
                             # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                             #     there is no way that we get LL contribution from this s, so skip it
         
         
                             if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                                 continue
                             end
         
                             # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                             # well I guess you could still call it influenced, just with value 0
         
                             if x[obs_idx, idx, loose_idx] ≈ 0
                                 push!(influenced_alts, obs_idx)
                             end
         
                             # so, now its influenced
         
                             # we do also need to add the best constant no? Only if there exists a constant one ofc
                             if !(max_j == -1) && !(max_j in influenced_alts)
                                 push!(influenced_alts, max_j)
                             end
         
                             # 3.) compute breakpoints: 
         
                             # compute constants
                             c_ia_idx = zeros(J)
         
                             # I mean this should be also computing the winning constant right?
         
                             for ia in influenced_alts
                                 c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[ia, idx]
                             end
         
                             # check for parallel slopes this would trigger if time_car = time_train for example
                             # mixed makes no difference to this
                             continue_flag = false
                             for ia in influenced_alts
                                 if ia != obs_idx
                                     if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                         if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                             # if osberved is parallel with another one and smaller constant then
                                             # we will never get that guys sLL
                                             continue_flag = true
                                             break 
                                         end
                                     end
                                 end
                             end
         
                             if continue_flag
                                 continue
                             end
         
                             # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                             # it could still have slope 0 though
         
                             # now, if there is only obs_idx and one other alternative in influenced_alts, 
                             # we can compute the breakpoint easily like this: 
         
                             # no you cant because there is still the best constant you have to take into account
                             # literally having only one influenced alt is the only scenario which can be treated distinctly
         
                             # length(influenced_alts) many affine lines mx + b
                             # c_ia_idx[ia] are the constant parts b
                             # x[ia, idx, loose_idx] are the coefficients m 
         
                             b_aff = c_ia_idx
         
                             m_aff = zeros(J)
                             for ia in influenced_alts
                                 m_aff[ia] = x[ia, idx, loose_idx]
                             end
         
                             seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)
         
                             # so this is either empty if there is no dominance segment
                             # or the full bounds if we dominate everywhere
                             # or one of the two could be bounds 
                             # or none
         
                             if isempty(seg_idx)
                                 # nothing to get here
                                 continue
                             end
         
                             if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                                 # a winner!
                                 hits[n] += 1
                                 continue
                             end
         
                             # by this point we know that not both bounds are BOUNDS
         
                             plusser = false
         
                             if seg_idx[1] == a[loose_idx]
                                 # so it starts at the lower bound and ends somewhere within the bounds
                                 # we call this a minusser
                                 hits[n] += 1
                                 plusser = false
                                 push!(cp, [seg_idx[2], n, idx, plusser])
                             elseif seg_idx[2] == b[loose_idx]
                                 # so it starts somewhere within the bounds and ends at the upper bound
                                 # we call this a plusser
                                 plusser = true
                                 push!(cp, [seg_idx[1], n, idx, plusser])
                             else
                                 # so it starts somewhere within the bounds and ends somewhere between the bounds
                                 # the beginning is thus a plusser and the end a minusser
                                 push!(cp, [seg_idx[1], n, idx, true])
                                 push!(cp, [seg_idx[2], n, idx, false])
                             end
                         end
            else # class 4
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_4_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                            
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end
                                
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end    
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_4_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end
                    
                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            end
        end
    else  # estimating latent class probabilites
        # compute the utility everyone gets with the different classes (and which one is the chosen in each case)
        for idx in 1:(NS)   
            av_idx = findall(av[:, idx] .== 1) 
            
            n = div(idx - 1, R) + 1
            winners_classes = []

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)
            
            for c in 1:4
                max_u = -Inf
                max_j = -1
                for j in av_idx
                    if c == 1
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                    elseif c == 2
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                    elseif c == 3
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                    else
                        u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                    end
                    # Track the maximum value and corresponding index
                    if u > max_u + toll
                        max_u = u
                        max_j = j
                    elseif abs(u - max_u) < toll && j == obs_idx
                        max_u = u
                        max_j = j
                    end
                end

                if max_j == obs_idx
                    push!(winners_classes, c)
                end
            end
            
            if winners_classes == [1, 2, 3, 4]
                hits[n] += 1
                # no matter the breakpoint we always get this gentleman
                continue
            end

            if winners_classes == []
                # no matter the breakpoint we always loose this notgentleman
                continue
            end

            if loose_idx == prob_inds[1] # this influences the separator from class 1 to 2
                if sigma[idx] > beta[prob_inds[2]] && sigma[idx] <= beta[prob_inds[3]]
                    if 3 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] > beta[prob_inds[2]] && sigma[idx] > beta[prob_inds[3]]
                    if 4 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else
                    if (1 in winners_classes) && !(2 in winners_classes) # only class 1 wins
                        # add breakpoint = sigma[idx] as a plusser, since as soon as beta[7] > sigma[idx], we put him in
                        # class 1 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (2 in winners_classes) && !(1 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[7] > sigma[idx], we put him in
                        # class 1 and thus loose him
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (1 in winners_classes) && (2 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(1 in winners_classes) && !(2 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            elseif loose_idx == prob_inds[2] # this influences the separator from class 2 to 3 
                if sigma[idx] <= beta[prob_inds[1]]
                    if 1 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] > beta[prob_inds[3]]
                    if 4 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else   
                    if (2 in winners_classes) && !(3 in winners_classes) # only class 2 wins
                        # add breakpoint = sigma[idx] as a plusser since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (3 in winners_classes) && !(2 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and we loose him 
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (2 in winners_classes) && (3 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(2 in winners_classes) && !(3 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            else # this influences the separator from class 3 to 4 
                if sigma[idx] <= beta[prob_inds[1]]
                    if 1 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] <= beta[prob_inds[2]]
                    if 2 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else   
                    if (3 in winners_classes) && !(4 in winners_classes) # only class 3 wins
                        # add breakpoint = sigma[idx] as a plusser
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (4 in winners_classes) && !(3 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (3 in winners_classes) && (4 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(3 in winners_classes) && !(4 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            end
        end
    end

    # end).time
    
    # println("building cp took $time_build_cp s")
    
    sLL_tot = - N * log(R) + sum((s = hits[n]; s > 0 ? log(s) : -100) for n in 1:N)    
    sLL_opt = - N * log(R) # this makes more sense no?
    beta_opt = copy(beta)
    
    # II) Test all the critical value of loose_idx
    
    # sorting
    sorted_cp = sort(cp, by = x -> x[1])  
    
    for xd in sorted_cp
        n = Int(xd[2])
        idx = Int(xd[3])
        m = hits[n]
        beta[loose_idx] = xd[1]

        if Bool(xd[4]) # if plusser
            if m == 0
                sLL_tot += log(m+1) - (-100)
                hits[n] += 1
            else
                sLL_tot += log(m+1) - log(m)
                hits[n] += 1 
            end
        else # if minusser
            # sunlight means immediate minuss procedure
            if m == 1 
                sLL_tot += (-100) - log(m)
                hits[n] -= 1
            else
                sLL_tot += log(m-1) - log(m)
                hits[n] -= 1
            end
        end

        if sLL_tot > sLL_opt
           sLL_opt = sLL_tot
           beta_opt = deepcopy(beta)
        end
    end
            
    # Optimal solution
    return beta_opt, sLL_opt
end;

function BHAMSLE_onedim_latent5classes(x, y, av, epsilon, sigma, R, beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, prob_inds, a, b)    
    # Problem size
    NS = size(epsilon)[2]
    J = size(x)[1]
    K = size(x)[3]
    N = Int(NS / R)

    #Initial solution
    beta = zeros(length(beta))
    logOfZero = -100
	iteration = 0
    
    # identify loose price AND define beta I guess
    loose_idx = findfirst(j -> a[j] != b[j], 1:length(a))
    
    for i in 1:length(a)
        if i != loose_idx
            beta[i] = a[i]
        end
    end
    
    # BoundsError: attempt to access 1-element Vector{Float64} at index [2]
    
    # compute breakpoints 
    cp = []
    
    # initialize the greatest hits
    hits = zeros(N)
    
    # time_build_cp = (@timed begin
    
    toll = 1e-6
    
    plusslack = -0.999e-6  # (tol - epsilon) this should be 0.999* the utility comparison tolerance
    # we changed its sign too so that we can just do the same stuff as we did for sunlight
    # without getting confused
    
    sunlight = -1.5e-6 # (tol + epsilon) set to 0 to activate shadow
    # funny story, we subtract the sunlight from obs, soo at the bp we get obs = compet + sunlight
    # so if we have sunlight positive thats wrong, the point is that obs is at least sunlight WORSE
    # which is why we have to make it negative
    
    # this determines the amount of sunlight. Should be ever so slightly more than the
    # sLL computation threshold of comparing utilties. Controls when a minusser is removed
    # (when the observed utility is "sunlight" away from best competition)
    
    if !(loose_idx in prob_inds) 
        #     if sigma < beta[7]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
        #     elseif sigma < beta[8]
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
        #     else
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
        #     end
        # end
        
        for idx in 1:(NS)   
            av_idx = findall(av[:, idx] .== 1) 
#             if !(sigma[idx] ≈ beta[7]) && !(sigma[idx] ≈ beta[8])
            # actually I decided that we wont bother with this nitpicking. Its a heuristic.
            if sigma[idx] <= beta[prob_inds[1]] # class 1
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_1_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                           if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                    
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                    # we had to do the bounds check first
                    
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end      
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                             if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_1_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            elseif sigma[idx] <= beta[prob_inds[2]] # class 2
       # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_2_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                        
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end


                    # 3.) compute breakpoint: 
                            
                    # lets think about. it if loose idx == 6, EVERY alt is influenced, since its time.
                    # then eeh yeah I mean you should divide by stressfactor too wtf. Then we dont need it in the constant LHS
                            

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                            
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end              
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_2_ks if !(k in [loose_idx])) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            elseif sigma[idx] <= beta[prob_inds[3]] # class 3
                # #     do all the checking with influenced alts and whatnot with this utility:
                 #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_3_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                    if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                        
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end


                    # 3.) compute breakpoint: 
                            
                    # lets think about. it if loose idx == 6, EVERY alt is influenced, since its time.
                    # then eeh yeah I mean you should divide by stressfactor too wtf. Then we dont need it in the constant LHS
                            

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                            
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end              
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_3_ks if !(k in [loose_idx])) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            elseif sigma[idx] <= beta[prob_inds[4]] # class 4
                # #     do all the checking with influenced alts and whatnot with this utility:
                 #         u = sum(x[j, idx, k] * beta[k] for k in class_1_ks if k != 6) + x[j, idx, 6] * stressfactor * beta[6] + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_4_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                    if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                        
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            # then max_u has to be the observed
                            # either influenced or max_u is best, else we would have gotten stopped
                            # at the condition right before
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end


                    # 3.) compute breakpoint: 
                            
                    # lets think about. it if loose idx == 6, EVERY alt is influenced, since its time.
                    # then eeh yeah I mean you should divide by stressfactor too wtf. Then we dont need it in the constant LHS
                            

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_4_ks if !(k in [loose_idx])) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end

                    # ok so here we now that the dude has a bp that is within the bounds
                            
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end              
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_4_ks if !(k in [loose_idx])) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end

                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            else # class 5
        # #     do all the checking with influenced alts and whatnot with this utility:
        #         u = sum(x[j, idx, k] * beta[k] for k in 1:K) + epsilon[j, idx]
                n = div(idx - 1, R) + 1
                
                if !(loose_idx in class_5_ks) # class 1 is unaffected by beta_RH
                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_5_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                       if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
                
                x_idx = x[:, idx, loose_idx]
                influenced_alts = findall(x -> x != 0, x_idx)

                if isempty(influenced_alts)
                    # if nothing is influenced by that loose thing, again it could be a case where the current idx
                    # for example is not part of that latent class. He still has utilities

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    # 1.) compute index and utility of best constant alternative     
                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute utilities, which are ALL constant
                    for j in av_idx
                        u = sum(x[j, idx, k] * beta[k] for k in class_5_ks) + epsilon[j, idx]
                        # Track the maximum value and corresponding index
                        if u > max_u + toll
                            max_u = u
                            max_j = j
                        elseif abs(u - max_u) < toll && j == obs_idx
                            max_u = u
                            max_j = j
                        end
                    end
                    # max_u and max_j are now what we were looking for

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if obs_idx == max_j
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                elseif length(influenced_alts) == 1
                    # so one influenced utility against the best constant one

                    # 1.) compute index and utility of best constant alternative   

                    influenced_alt = influenced_alts[1]

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities
                    for j in av_idx
                        if j != influenced_alt
                            u = sum(x[j, idx, k] * beta[k] for k in class_5_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                   # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it

                    if (obs_idx != max_j) && (obs_idx != influenced_alt)
                        continue
                    end
                            
                    # is the individual a plusser or a minusser?
                    plusser = true                    
                    # you're a minusser if:
                    if x[influenced_alt, idx, loose_idx] < 0 
                        if obs_idx == influenced_alt 
                            plusser = false
                            breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in class_5_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else                        
                            breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in class_5_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    else
                        if obs_idx == influenced_alt
                            breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_5_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        else
                            plusser = false
                            breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in class_5_ks if k != loose_idx) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                        end
                    end

                    # we have to check if that breakpoint is within the bounds

                    if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                        # then we should either count it as a hit or not, but in both cases skip this idx
                        # we can choose any point in a,b to evaluate whether its a safe one or not
                        # and again it depends on whether or not the observed is the constant or not

                        infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in class_5_ks if k != loose_idx) + epsilon[influenced_alt, idx]

                        if obs_idx == influenced_alt # observed is not constant
                            if max_u <= infl_u
                                # so the dependent is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        else # observed is the best constant
                            if max_u >= infl_u
                                # so the constant is at least as good, we count it as a win
                                hits[n] += 1
                                continue
                            else
                                continue
                            end
                        end
                    end
                                
                    if plusser
                        push!(cp, [breakpoint, n, idx, plusser])
                    else
                        hits[n] += 1
                        push!(cp, [breakpoint, n, idx, plusser])
                    end    
                else    
                    # now we are in the interesting case, having multiple influenced alts

                    # 1.) compute index and utility of best constant alternative   

                    # max_j is the index of the constant alternative, and max_u is its utility value

                    y_idx = y[:, idx]
                    obs_idx = findfirst(x -> x == 1, y_idx)

                    max_u = -Inf   # Initialize max value (assuming u contains real numbers)
                    max_j = -1     # Initialize index of max value

                    # compute all constant utilities

                    for j in av_idx
                        if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                            u = sum(x[j, idx, k] * beta[k] for k in class_5_ks) + epsilon[j, idx]
                            # Track the maximum value and corresponding index
                            if u > max_u + toll
                                max_u = u
                                max_j = j
                            elseif abs(u - max_u) < toll && j == obs_idx
                                max_u = u
                                max_j = j
                            end
                        end
                    end

                    # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
                    #     there is no way that we get LL contribution from this s, so skip it


                    if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                        continue
                    end

                    # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
                    # well I guess you could still call it influenced, just with value 0

                    if x[obs_idx, idx, loose_idx] ≈ 0
                        push!(influenced_alts, obs_idx)
                    end

                    # so, now its influenced

                    # we do also need to add the best constant no? Only if there exists a constant one ofc
                    if !(max_j == -1) && !(max_j in influenced_alts)
                        push!(influenced_alts, max_j)
                    end

                    # 3.) compute breakpoints: 

                    # compute constants
                    c_ia_idx = zeros(J)

                    # I mean this should be also computing the winning constant right?

                    for ia in influenced_alts
                        c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in class_5_ks if k != loose_idx) + epsilon[ia, idx]
                    end

                    # check for parallel slopes this would trigger if time_car = time_train for example
                    # mixed makes no difference to this
                    continue_flag = false
                    for ia in influenced_alts
                        if ia != obs_idx
                            if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                                if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                    # if osberved is parallel with another one and smaller constant then
                                    # we will never get that guys sLL
                                    continue_flag = true
                                    break 
                                end
                            end
                        end
                    end

                    if continue_flag
                        continue
                    end

                    # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
                    # it could still have slope 0 though

                    # now, if there is only obs_idx and one other alternative in influenced_alts, 
                    # we can compute the breakpoint easily like this: 

                    # no you cant because there is still the best constant you have to take into account
                    # literally having only one influenced alt is the only scenario which can be treated distinctly

                    # length(influenced_alts) many affine lines mx + b
                    # c_ia_idx[ia] are the constant parts b
                    # x[ia, idx, loose_idx] are the coefficients m 

                    b_aff = c_ia_idx

                    m_aff = zeros(J)
                    for ia in influenced_alts
                        m_aff[ia] = x[ia, idx, loose_idx]
                    end
                    
                    seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

                    # so this is either empty if there is no dominance segment
                    # or the full bounds if we dominate everywhere
                    # or one of the two could be bounds 
                    # or none

                    if isempty(seg_idx)
                        # nothing to get here
                        continue
                    end

                    if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                        # a winner!
                        hits[n] += 1
                        continue
                    end

                    # by this point we know that not both bounds are BOUNDS

                    plusser = false

                    if seg_idx[1] == a[loose_idx]
                        # so it starts at the lower bound and ends somewhere within the bounds
                        # we call this a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [seg_idx[2], n, idx, plusser])
                    elseif seg_idx[2] == b[loose_idx]
                        # so it starts somewhere within the bounds and ends at the upper bound
                        # we call this a plusser
                        plusser = true
                        push!(cp, [seg_idx[1], n, idx, plusser])
                    else
                        # so it starts somewhere within the bounds and ends somewhere between the bounds
                        # the beginning is thus a plusser and the end a minusser
                        push!(cp, [seg_idx[1], n, idx, true])
                        push!(cp, [seg_idx[2], n, idx, false])
                    end
                end
            end
        end
    else  # estimating latent class probabilites
        # compute the utility everyone gets with the different classes (and which one is the chosen in each case)
        for idx in 1:(NS)   
            av_idx = findall(av[:, idx] .== 1) 
            
            n = div(idx - 1, R) + 1
            winners_classes = []

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)
            
            for c in 1:5
                max_u = -Inf
                max_j = -1
                for j in av_idx
                    if c == 1
                        u = sum(x[j, idx, k] * beta[k] for k in class_1_ks) + epsilon[j, idx]
                    elseif c == 2
                        u = sum(x[j, idx, k] * beta[k] for k in class_2_ks) + epsilon[j, idx]
                    elseif c == 3
                        u = sum(x[j, idx, k] * beta[k] for k in class_3_ks) + epsilon[j, idx]
                    elseif c == 4
                        u = sum(x[j, idx, k] * beta[k] for k in class_4_ks) + epsilon[j, idx]
                    else
                        u = sum(x[j, idx, k] * beta[k] for k in class_5_ks) + epsilon[j, idx]
                    end
                    # Track the maximum value and corresponding index
                    if u > max_u + toll
                        max_u = u
                        max_j = j
                    elseif abs(u - max_u) < toll && j == obs_idx
                        max_u = u
                        max_j = j
                    end
                end

                if max_j == obs_idx
                    push!(winners_classes, c)
                end
            end
            
            if winners_classes == [1, 2, 3, 4, 5]
                hits[n] += 1
                # no matter the breakpoint we always get this gentleman
                continue
            end

            if winners_classes == []
                # no matter the breakpoint we always loose this notgentleman
                continue
            end

            if loose_idx == prob_inds[1] # this influences the separator from class 1 to 2
                if sigma[idx] > beta[prob_inds[2]] && sigma[idx] <= beta[prob_inds[3]]
                    if 3 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] > beta[prob_inds[2]] && sigma[idx] > beta[prob_inds[3]]
                    if 4 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else
                    if (1 in winners_classes) && !(2 in winners_classes) # only class 1 wins
                        # add breakpoint = sigma[idx] as a plusser, since as soon as beta[7] > sigma[idx], we put him in
                        # class 1 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (2 in winners_classes) && !(1 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[7] > sigma[idx], we put him in
                        # class 1 and thus loose him
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (1 in winners_classes) && (2 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(1 in winners_classes) && !(2 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            elseif loose_idx == prob_inds[2] # this influences the separator from class 2 to 3 
                if sigma[idx] <= beta[prob_inds[1]]
                    if 1 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] > beta[prob_inds[3]]
                    if 4 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else   
                    if (2 in winners_classes) && !(3 in winners_classes) # only class 2 wins
                        # add breakpoint = sigma[idx] as a plusser since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (3 in winners_classes) && !(2 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and we loose him 
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (2 in winners_classes) && (3 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(2 in winners_classes) && !(3 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            elseif loose_idx == prob_inds[3] # this influences the separator from class 3 to 4 
                if sigma[idx] <= beta[prob_inds[1]]
                    if 1 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] <= beta[prob_inds[2]]
                    if 2 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] > beta[prob_inds[4]]
                    if 5 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else   
                    if (3 in winners_classes) && !(4 in winners_classes) # only class 3 wins
                        # add breakpoint = sigma[idx] as a plusser since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and thus we win
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (4 in winners_classes) && !(3 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser, since as soon as beta[8] > sigma[idx], we put him in
                        # class 2 and we loose him 
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (3 in winners_classes) && (4 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(3 in winners_classes) && !(4 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            else # this influences the separator from class 4 to 5 
                if sigma[idx] <= beta[prob_inds[1]]
                    if 1 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] <= beta[prob_inds[2]]
                    if 2 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                elseif sigma[idx] <= beta[prob_inds[3]]
                    if 3 in winners_classes
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    else
                        continue # it means we are guaranteed in a loosers class, nothing we can do
                    end
                else   
                    if (4 in winners_classes) && !(5 in winners_classes) # only class 4 wins
                        # add breakpoint = sigma[idx] as a plusser
                        plusser = true
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (5 in winners_classes) && !(4 in winners_classes)
                        # add breakpoint = sigma[idx] as a minusser
                        hits[n] += 1
                        plusser = false
                        push!(cp, [sigma[idx], n, idx, plusser])
                    elseif (4 in winners_classes) && (5 in winners_classes)
                        hits[n] += 1
                        # no matter the breakpoint we always get this gentleman
                        continue
                    elseif !(4 in winners_classes) && !(5 in winners_classes)
                        continue # no matter what we always loose
                    end
                end
            end
        end
    end

    # end).time
    
    # println("building cp took $time_build_cp s")
    
    sLL_tot = - N * log(R) + sum((s = hits[n]; s > 0 ? log(s) : -100) for n in 1:N)    
    sLL_opt = - N * log(R) # this makes more sense no?
    beta_opt = copy(beta)
    
    # II) Test all the critical value of loose_idx
    
    # sorting
    sorted_cp = sort(cp, by = x -> x[1])  
    
    for xd in sorted_cp
        n = Int(xd[2])
        idx = Int(xd[3])
        m = hits[n]
        beta[loose_idx] = xd[1]

        if Bool(xd[4]) # if plusser
            if m == 0
                sLL_tot += log(m+1) - (-100)
                hits[n] += 1
            else
                sLL_tot += log(m+1) - log(m)
                hits[n] += 1 
            end
        else # if minusser
            # sunlight means immediate minuss procedure
            if m == 1 
                sLL_tot += (-100) - log(m)
                hits[n] -= 1
            else
                sLL_tot += log(m-1) - log(m)
                hits[n] -= 1
            end
        end

        if sLL_tot > sLL_opt
           sLL_opt = sLL_tot
           beta_opt = deepcopy(beta)
        end
    end
            
    # Optimal solution
    return beta_opt, sLL_opt
end;


function BHAMSLE_onedim_mixed(x, y, av, epsilon, sigma, R, beta, mix_inds, a, b)  
    # Problem size
    NS = size(epsilon)[2]
    J = size(x)[1]
    K = size(x)[3]
    N = Int(NS / R)

    #Initial solution
    beta = zeros(length(beta))
    logOfZero = -100
	iteration = 0
    
    # identify loose price AND define beta I guess
    loose_idx = findfirst(j -> a[j] != b[j], 1:length(a))
    
    for i in 1:length(a)
        if i != loose_idx
            beta[i] = a[i]
        end
    end
    
    # compute breakpoints 
    cp = []
    hits = zeros(N)
    
    toll = 1e-6
    
    plusslack = -0.999e-6  # (tol - epsilon) this should be 0.999* the utility comparison tolerance
    sunlight = -1.5e-6 # (tol + epsilon) set to 0 to activate shadow

    # sigma should ne H N R, where ofc H is the number of mixed parameters
    # or eeeeh it could be HxNR bien sure, maybe more likely

    # this should be useful:

    cursies = [k[2] for k in mix_inds]

    if loose_idx in cursies # cursed
        cursed_idx = findfirst(x -> x[2] == loose_idx, mix_inds)
        cursed_first = mix_inds[cursed_idx][1]
    else
        cursed_idx = nothing
        cursed_first = nothing
    end
	
    for idx in 1:(NS)   
        av_idx = findall(av[:, idx] .== 1)
        n = div(idx - 1, R) + 1
		if loose_idx in cursies 
        	x_idx = x[:, idx, cursed_first]
		else
			x_idx = x[:, idx, loose_idx]
		end
        influenced_alts = findall(x -> x != 0, x_idx)

        if isempty(influenced_alts)
            # if nothing is influenced by that loose thing, again it could be a case where the current idx
            # for example is not part of that latent class. 
            
            # Orrrrr he just has the x = 0 there. In optima someone might have trip purpose other, and the 
            # loose beta corr. to beta_cost_hwh you feel me. 
            
            # anyway: He still has utilities

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)

            # 1.) compute index and utility of best constant alternative     
            max_u = -Inf   # Initialize max value (assuming u contains real numbers)
            max_j = -1     # Initialize index of max value

            # compute utilities, which are ALL constant
            for j in av_idx
                u = sum(x[j, idx, k] * beta[k] for k in 1:K) + sum(x[j, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[j, idx]
                # Track the maximum value and corresponding index
                if u > max_u + toll
                    max_u = u
                    max_j = j
                elseif abs(u - max_u) < toll && j == obs_idx
                    max_u = u
                    max_j = j
                end
            end
            # max_u and max_j are now what we were looking for

            # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
            #     there is no way that we get LL contribution from this s, so skip it

            if obs_idx == max_j
                hits[n] += 1
                continue
            else
                continue
            end

        elseif length(influenced_alts) == 1
            # so one influenced utility against the best constant one

            # 1.) compute index and utility of best constant alternative   

            influenced_alt = influenced_alts[1]

            # max_j is the index of the constant alternative, and max_u is its utility value

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)

            max_u = -Inf   # Initialize max value (assuming u contains real numbers)
            max_j = -1     # Initialize index of max value

            # compute all constant utilities
            for j in av_idx
                if j != influenced_alt
                    u = sum(x[j, idx, k] * beta[k] for k in 1:K) + sum(x[j, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[j, idx]
                    # Track the maximum value and corresponding index
                    if u > max_u + toll
                        max_u = u
                        max_j = j
                    elseif abs(u - max_u) < toll && j == obs_idx
                        max_u = u
                        max_j = j
                    end
                end
            end

            # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
            #     there is no way that we get LL contribution from this s, so skip it

            if (obs_idx != max_j) && (obs_idx != influenced_alt)
                continue
            end
            

            # lets say loose idx is beta_mean for cost_hwh
            # it multiplies cost, which is pos
            # yeye actually all good, as long as we are not in the cursed indices

            if !(loose_idx in cursies) # not a cursed index
                # is the individual a plusser or a minusser?
                plusser = true                    
                # you're a minusser if:
                if x[influenced_alt, idx, loose_idx] < 0
                    if obs_idx == influenced_alt 
                        plusser = false
                        breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K if k != loose_idx) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                    else                        
                        breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K if k != loose_idx) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                    end
                else
                    if obs_idx == influenced_alt
                        breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K if k != loose_idx) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                    else
                        plusser = false
                        # then max_u has to be the observed
                        # either influenced or max_u is best, else we would have gotten stopped
                        # at the condition right before
                        breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K if k != loose_idx) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[influenced_alt, idx])) / x[influenced_alt, idx, loose_idx]
                    end
                end
            else # cursed index
                # is the individual a plusser or a minusser?
                plusser = true                    
                # you're a minusser if:
                if x[influenced_alt, idx, cursed_first] < 0
                    if obs_idx == influenced_alt 
                        plusser = false
                        breakpoint = (max_u - (- sunlight + sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds) if k[2] != loose_idx; init=0.0) + epsilon[influenced_alt, idx])) / (x[influenced_alt, idx, cursed_first] * sigma[cursed_idx, idx])
                    else                        
                        breakpoint = (max_u - (- plusslack + sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds) if k[2] != loose_idx; init=0.0) + epsilon[influenced_alt, idx])) / (x[influenced_alt, idx, cursed_first] * sigma[cursed_idx, idx])
                    end
                else
                    if obs_idx == influenced_alt
                        breakpoint = ((max_u - plusslack) - (sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds) if k[2] != loose_idx; init=0.0) + epsilon[influenced_alt, idx])) / (x[influenced_alt, idx, cursed_first] * sigma[cursed_idx, idx])
                    else
                        plusser = false
                        # then max_u has to be the observed
                        # either influenced or max_u is best, else we would have gotten stopped
                        # at the condition right before
                        breakpoint = ((max_u - sunlight) - (sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds) if k[2] != loose_idx; init=0.0) + epsilon[influenced_alt, idx])) / (x[influenced_alt, idx, cursed_first] * sigma[cursed_idx, idx])
                    end
                end
            end

            # we have to check if that breakpoint is within the bounds

            if !(a[loose_idx] <= breakpoint <= b[loose_idx])
                # then we should either count it as a hit or not, but in both cases skip this idx
                # we can choose any point in a,b to evaluate whether its a safe one or not
                # and again it depends on whether or not the observed is the constant or not
                if !(loose_idx in cursies) # not a cursed index
                    infl_u = x[influenced_alt, idx, loose_idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k] * beta[k] for k in 1:K if k != loose_idx) + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[influenced_alt, idx]
                else
                    infl_u = x[influenced_alt, idx, cursed_first] * sigma[cursed_idx, idx] * (a[loose_idx] + b[loose_idx]) / 2 + sum(x[influenced_alt, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds) if k[2] != loose_idx; init=0.0) + epsilon[influenced_alt, idx]
                end
                if obs_idx == influenced_alt # observed is not constant
                    if max_u <= infl_u
                        # so the dependent is at least as good, we count it as a win
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                else # observed is the best constant
                    if max_u >= infl_u
                        # so the constant is at least as good, we count it as a win
                        hits[n] += 1
                        continue
                    else
                        continue
                    end
                end
            end

            # ok so here we now that the dude has a bp that is within the bounds            
            if plusser
                push!(cp, [breakpoint, n, idx, plusser])
            else
                hits[n] += 1
                push!(cp, [breakpoint, n, idx, plusser])
            end      
        else    
            # now we are in the interesting case, having multiple influenced alts

            # 1.) compute index and utility of best constant alternative   

            # max_j is the index of the constant alternative, and max_u is its utility value

            y_idx = y[:, idx]
            obs_idx = findfirst(x -> x == 1, y_idx)

            max_u = -Inf   # Initialize max value (assuming u contains real numbers)
            max_j = -1     # Initialize index of max value

            # compute all constant utilities

            for j in av_idx
                if !(j in influenced_alts) # if mixed beta was loose, its set to 0 anyway
                    u = sum(x[j, idx, k] * beta[k] for k in 1:K) + sum(x[j, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds)) + epsilon[j, idx]
                    # Track the maximum value and corresponding index
                        if u > max_u + toll
                        max_u = u
                        max_j = j
                    elseif abs(u - max_u) < toll && j == obs_idx
                        max_u = u
                        max_j = j
                    end
                end
            end

            # 2.) if the observed alternative is neither the best constant one nor influenced by the loose idx beta, 
            #     there is no way that we get LL contribution from this s, so skip it


            if (obs_idx != max_j) && !(obs_idx in influenced_alts)
                continue
            end

            # wait, so after this the observed idx could actually be NOT influenced, it would be the constant
            # well I guess you could still call it influenced, just with value 0
			
			if loose_idx in cursies 
	            if x[obs_idx, idx, cursed_first] ≈ 0
	                push!(influenced_alts, obs_idx)
	            end
			else
	            if x[obs_idx, idx, loose_idx] ≈ 0
	                push!(influenced_alts, obs_idx)
	            end
			end

            # so, now its influenced

            # we do also need to add the best constant no? Only if there exists a constant one ofc
            if !(max_j == -1) && !(max_j in influenced_alts)
                push!(influenced_alts, max_j)
            end

            # 3.) now compute breakpoints: 

            # compute constants
            c_ia_idx = zeros(J)

            # I mean this should be also computing the winning constant right?
			if loose_idx in cursies && length(mix_inds) == 1
				# if there is only one mixed parameter and its the loose idx, the sigma part disappears fully
	            for ia in influenced_alts
	                c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in 1:K if k != loose_idx) + epsilon[ia, idx]
	            end
			else
	            for ia in influenced_alts
	                c_ia_idx[ia] = sum(x[ia, idx, k] * beta[k] for k in 1:K if k != loose_idx) + sum(x[ia, idx, k[1]] * sigma[h, idx] * beta[k[2]] for (h, k) in enumerate(mix_inds) if k[2] != loose_idx; init=0.0) + epsilon[ia, idx]
	            end
			end
			


            # check for parallel slopes this would trigger if time_car = time_train for example
            # mixed makes no difference to this
            continue_flag = false
            for ia in influenced_alts
                if ia != obs_idx
                    if loose_idx in cursies
                        if x[obs_idx, idx, cursed_first] ≈ x[ia, idx, cursed_first]
                            if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                # if osberved is parallel with another one and smaller constant then
                                # we will never get that guys sLL
                                continue_flag = true
                                break 
                            end
                        end
                    else
                        if x[obs_idx, idx, loose_idx] ≈ x[ia, idx, loose_idx]
                            if c_ia_idx[ia] > c_ia_idx[obs_idx]
                                # if osberved is parallel with another one and smaller constant then
                                # we will never get that guys sLL
                                continue_flag = true
                                break 
                            end
                        end
                    end


                end
            end

            if continue_flag
                continue
            end

            # otherwise we know that eeh at least the obs_idx is not parallel to any of the other ones
            # it could still have slope 0 though

            # now, if there is only obs_idx and one other alternative in influenced_alts, 
            # we can compute the breakpoint easily like this: 

            # no you cant because there is still the best constant you have to take into account
            # literally having only one influenced alt is the only scenario which can be treated distinctly

            # length(influenced_alts) many affine lines mx + b
            # c_ia_idx[ia] are the constant parts b
            # x[ia, idx, loose_idx] are the coefficients m 

            b_aff = c_ia_idx

            m_aff = zeros(J)
            for ia in influenced_alts
                if loose_idx in cursies
                    m_aff[ia] = x[ia, idx, cursed_first] * sigma[cursed_idx, idx]
                else
                    m_aff[ia] = x[ia, idx, loose_idx]
                end
            end

            seg_idx = compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)

            # so this is either empty if there is no dominance segment
            # or the full bounds if we dominate everywhere
            # or one of the two could be bounds 
            # or none

            if isempty(seg_idx)
                # nothing to get here
                continue
            end

            if seg_idx[1] == a[loose_idx] && seg_idx[2] == b[loose_idx]
                # a winner!
                hits[n] += 1
                continue
            end

            # by this point we know that not both bounds are BOUNDS

            plusser = false

            if seg_idx[1] == a[loose_idx]
                # so it starts at the lower bound and ends somewhere within the bounds
                # we call this a minusser
                hits[n] += 1
                plusser = false
                push!(cp, [seg_idx[2], n, idx, plusser])
            elseif seg_idx[2] == b[loose_idx]
                # so it starts somewhere within the bounds and ends at the upper bound
                # we call this a plusser
                plusser = true
                push!(cp, [seg_idx[1], n, idx, plusser])
            else
                # so it starts somewhere within the bounds and ends somewhere between the bounds
                # the beginning is thus a plusser and the end a minusser
                push!(cp, [seg_idx[1], n, idx, true])
                push!(cp, [seg_idx[2], n, idx, false])
            end
        end
    end
    
    sLL_tot = - N * log(R) + sum((s = hits[n]; s > 0 ? log(s) : -100) for n in 1:N)    
    sLL_opt = - N * log(R) # this makes more sense no?
    beta_opt = copy(beta)
    
    # II) Test all the critical value of loose_idx
    
    # sorting
    sorted_cp = sort(cp, by = x -> x[1])  
	
    for xd in sorted_cp
        n = Int(xd[2])
        idx = Int(xd[3])
        m = hits[n]
        beta[loose_idx] = xd[1]

        if Bool(xd[4]) # if plusser
            if m == 0
                sLL_tot += log(m+1) - (-100)
                hits[n] += 1
            else
                sLL_tot += log(m+1) - log(m)
                hits[n] += 1 
            end
        else # if minusser
            # sunlight means immediate minuss procedure
            if m == 1 
                sLL_tot += (-100) - log(m)
                hits[n] -= 1
            else
                sLL_tot += log(m-1) - log(m)
                hits[n] -= 1
            end
        end

        if sLL_tot > sLL_opt
           sLL_opt = sLL_tot
           beta_opt = deepcopy(beta)
        end
    end
            
    # Optimal solution
    return beta_opt, sLL_opt
end;



function compute_highest_segment_latent(m_aff, b_aff, influenced_alts, obs_idx, loose_idx, a, b, sunlight, plusslack)
    # Extract observed slope and intercept
    m_obs = m_aff[obs_idx]
    b_obs = b_aff[obs_idx]

    # Initialize segment as the full bound
    lower_bound = a[loose_idx]
    upper_bound = b[loose_idx]
    
    # Iterate over all other lines in influenced_alts
    for i in influenced_alts
        if i == obs_idx
            continue  # Skip the observed line
        end

        # Extract slope and intercept of the current line
        m_i = m_aff[i]
        b_i = b_aff[i]

        # If slopes are equal, check intercepts to rule out any intersections
        if m_obs == m_i
            if b_obs <= b_i  # If observed line is not strictly higher
                return []  # No valid segment
            end
            # If the intercept of the observed line is higher, continue as there's no intersection
        else
            # Compute intersection point between the observed line and the current line
#             beta_intersect = (b_i - b_obs) / (m_obs - m_i)
            # Check if the intersection lies within the current bounds
            if m_obs > m_i
                # Observed line is decreasing slower or increasing faster. So dominance continues after. 
                # So it's a segment start. So we compare to the lower bound. 
                # PLUSSER
                
                beta_intersect = (b_i - (b_obs - plusslack)) / (m_obs - m_i)
                
                if beta_intersect > lower_bound
                    lower_bound = beta_intersect
                end
                
            else
                # Observed line is increasing slower or decreasing faster. So dominance ends after. 
                # So it's a segment ending. So we compare to the upper bound.
                # MINUSSER
                
                beta_intersect = (b_i - (b_obs - sunlight)) / (m_obs - m_i)
                
                if beta_intersect < upper_bound
                    upper_bound = beta_intersect
                end
            end

            # If the segment becomes invalid, return []
            if lower_bound >= upper_bound
                return []
            end
        end
    end

    # Return the valid segment
    return [lower_bound, upper_bound]
end;

function BHAMSLE_mixed(x, y, av, epsilon, sigma, R, start_beta, mix_inds, a, b, beta_acc=1e-6, bp_proximity=1e-9)
    # initialize prices
    K = size(x)[3]
    start_beta = Float64.(start_beta) # cast prices to float in case the start prices were all integer
    beta = copy(start_beta)

    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1

    best_beta = copy(beta)
    consecutive_close = 0

    # Define a threshold for convergence
    conv_thresh = 1e-6
    iter = 0 # failsafe to not run forever in case of non-convergence

    _, _, sLL = compute_sLL_mixed(x, y, av, epsilon, R, sigma, beta, mix_inds)

    best_obj = sLL
    old_obj = sLL
    obj_val = best_obj
    old_beta = copy(best_beta)

    # println("Starting beta = $([round(b, digits=2) for b in beta]) gives objective value = ", sLL)
    
    K_here = K + length(mix_inds)
    
    while consecutive_close < K_here && iter < 5000
        iter += 1      
        
        result = @timed loose_beta, obj_val = do_BHAMSLE_onedim_mixed(x, y, av, epsilon, sigma, R, beta, loose_index, mix_inds, [a[loose_index]], [b[loose_index]], bp_proximity) 
        # Update the price at the current index
        
        beta[loose_index] = loose_beta
		
		# println("iter = $iter, beta = $([round(b, digits=2) for b in beta]) gives objective value = ", obj_val)
        
        # Check if the current obj_val and beta is very close to the previous one
        if abs(obj_val - old_obj) < conv_thresh
            consecutive_close += 1 # Increment the counter
        else
            beta_close = true
            for k in 1:K_here
                if abs(old_beta[k] - beta[k]) > beta_acc
#                     println("index $k. old_beta[$k] = $(old_beta[k]) != $(beta[k]) = beta[$k], so FALSE")
                    beta_close = false
                else
#                     println("index $k. old_beta[$k] = $(old_beta[k]) == $(beta[k]) = beta[$k], so true")
                end
            end
#             println("finally, beta_close = $beta_close")
            if beta_close
                consecutive_close += 1
#                 println("thus cc increases")
            else
                consecutive_close = 0  # Reset consecutive_close counter
#                 println("thus cc = 0")
            end
        end

        if obj_val > best_obj
            best_obj = obj_val
            best_beta = copy(beta)
            
        end 

        old_obj = copy(obj_val)
        old_beta = copy(beta)

        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > K_here
            loose_index = 1
        end

    end
    
    return best_obj, best_beta, iter
end;

function BHAMSLE_latent3classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, prob_inds, a, b, class_1_av, class_2_av, class_3_av, beta_acc=1e-6, bp_proximity=1e-9; extra_inds=nothing)
    # initialize prices
    K = size(x)[3]
    start_beta = Float64.(start_beta) # cast prices to float in case the start prices were all integer
    beta = copy(start_beta)

    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1

    best_beta = copy(beta)
    consecutive_close = 0

    # Define a threshold for convergence (for the obj value to count as close)
    conv_thresh = 1e-6
	
	# beta_acc = 1e-9 # this should maybe also be changed
	
    iter = 0 # failsafe to not run forever in case of non-convergence

    # _, _, sLL = compute_sLL_latent3classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks, class_3_ks)
    sLL = -Inf

    best_obj = sLL
    old_obj = sLL
    obj_val = best_obj
    old_beta = copy(best_beta)

#     println("Starting beta = $([round(b, digits=2) for b in beta]) gives objective value = ", sLL)
    
    K_here = prob_inds[end]
    
    while consecutive_close < K_here && iter < 5000
        iter += 1      
        
        result = @timed loose_beta, obj_val = do_BHAMSLE_onedim_latent3classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, class_3_ks, prob_inds, [a[loose_index]], [b[loose_index]], bp_proximity, class_1_av, class_2_av, class_3_av) 
        # Update the price at the current index
        
        if loose_index in prob_inds && loose_beta <= 1e-12
            nothing # dont update beta if there were no breakpoints found for probabilites (can actually happen)
        else
            beta[loose_index] = loose_beta
        end
        
        # Check if the current obj_val and beta is very close to the previous one
        if abs(obj_val - old_obj) < conv_thresh
            consecutive_close += 1 # Increment the counter
        else
            beta_close = true
            for k in 1:K_here
                if abs(old_beta[k] - beta[k]) > beta_acc
#                     println("index $k. old_beta[$k] = $(old_beta[k]) != $(beta[k]) = beta[$k], so FALSE")
                    beta_close = false
                else
#                     println("index $k. old_beta[$k] = $(old_beta[k]) == $(beta[k]) = beta[$k], so true")
                end
            end
#             println("finally, beta_close = $beta_close")
            if beta_close
                consecutive_close += 1
#                 println("thus cc increases")
            else
                consecutive_close = 0  # Reset consecutive_close counter
#                 println("thus cc = 0")
            end
        end

        if obj_val > best_obj
            best_obj = obj_val
            best_beta = copy(beta)
            
        end 

        old_obj = copy(obj_val)
        old_beta = copy(beta)

        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > K_here
            loose_index = 1
        end

    end
    
    return best_obj, best_beta, iter
end;


function BHAMSLE_latent2classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, prob_inds, a, b, class_1_av, class_2_av, beta_acc=1e-6, bp_proximity=1e-9)
    # initialize prices
    K = size(x)[3]
    start_beta = Float64.(start_beta) # cast prices to float in case the start prices were all integer
    beta = copy(start_beta)

    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1

    best_beta = copy(beta)
    consecutive_close = 0

    # Define a threshold for convergence
    conv_thresh = 1e-6
    iter = 0 # failsafe to not run forever in case of non-convergence

   #  _, _, sLL = compute_sLL_latent2classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks)
    sLL = -Inf 

    best_obj = sLL
    old_obj = sLL
    obj_val = best_obj
    old_beta = copy(best_beta)

#     println("Starting beta = $([round(b, digits=2) for b in beta]) gives objective value = ", sLL)
    K_here = prob_inds[end]
    
    while consecutive_close < K_here && iter < 5000
        iter += 1      
        
        result = @timed loose_beta, obj_val = do_BHAMSLE_onedim_latent2classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, prob_inds, [a[loose_index]], [b[loose_index]], bp_proximity, class_1_av, class_2_av) 
        # Update the price at the current index
        
        if loose_index in prob_inds && loose_beta <= 1e-12
            nothing # dont update beta if there were no breakpoints found for probabilites (can actually happen)
        else
            beta[loose_index] = loose_beta
        end
        
        # Check if the current obj_val and beta is very close to the previous one
        if abs(obj_val - old_obj) < conv_thresh
            consecutive_close += 1 # Increment the counter
        else
            beta_close = true
            for k in 1:K_here
                if abs(old_beta[k] - beta[k]) > beta_acc
#                     println("index $k. old_beta[$k] = $(old_beta[k]) != $(beta[k]) = beta[$k], so FALSE")
                    beta_close = false
                else
#                     println("index $k. old_beta[$k] = $(old_beta[k]) == $(beta[k]) = beta[$k], so true")
                end
            end
#             println("finally, beta_close = $beta_close")
            if beta_close
                consecutive_close += 1
#                 println("thus cc increases")
            else
                consecutive_close = 0  # Reset consecutive_close counter
#                 println("thus cc = 0")
            end
        end

        if obj_val > best_obj
            best_obj = obj_val
            best_beta = copy(beta)
            
        end 

        old_obj = copy(obj_val)
        old_beta = copy(beta)

        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > K_here
            loose_index = 1
        end

    end
    
    return best_obj, best_beta, iter
end;

function BHAMSLE_latent4classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, prob_inds, a, b, beta_acc=1e-6, bp_proximity=1e-9)
    # initialize prices
    K = size(x)[3]
    start_beta = Float64.(start_beta) # cast prices to float in case the start prices were all integer
    beta = copy(start_beta)

    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1

    best_beta = copy(beta)
    consecutive_close = 0

    # Define a threshold for convergence
    conv_thresh = 1e-6
    iter = 0 # failsafe to not run forever in case of non-convergence

    _, _, sLL = compute_sLL_latent4classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks, class_3_ks, class_4_ks)

    best_obj = sLL
    old_obj = sLL
    obj_val = best_obj
    old_beta = copy(best_beta)

    K_here = K + 3 # plus number of class parameter estimates
    
    while consecutive_close < K_here && iter < 5000
        iter += 1        
        result = @timed loose_beta, obj_val = do_BHAMSLE_onedim_latent4classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, class_3_ks, class_4_ks, prob_inds, [a[loose_index]], [b[loose_index]], bp_proximity) 

        if loose_index in prob_inds && loose_beta <= 1e-12
            nothing # dont update beta if there were no breakpoints found for probabilites (can actually happen)
        else
            beta[loose_index] = loose_beta
        end
        
        # Check if the current obj_val and beta is very close to the previous one
        if abs(obj_val - old_obj) < conv_thresh
            consecutive_close += 1 # Increment the counter
        else
            beta_close = true
            for k in 1:K_here
                if abs(old_beta[k] - beta[k]) > beta_acc
#                     println("index $k. old_beta[$k] = $(old_beta[k]) != $(beta[k]) = beta[$k], so FALSE")
                    beta_close = false
                else
#                     println("index $k. old_beta[$k] = $(old_beta[k]) == $(beta[k]) = beta[$k], so true")
                end
            end
            if beta_close
                consecutive_close += 1
            else
                consecutive_close = 0  # Reset consecutive_close counter
            end
        end

        if obj_val > best_obj
            best_obj = obj_val
            best_beta = copy(beta)
        end 

        old_obj = copy(obj_val)
        old_beta = copy(beta)

        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > K_here
            loose_index = 1
        end
        
    end
    
    return best_obj, best_beta, iter
end;

function BHAMSLE_latent5classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, prob_inds, a, b, beta_acc=1e-6, bp_proximity=1e-9)
    # initialize prices
    K = size(x)[3]
    start_beta = Float64.(start_beta) # cast prices to float in case the start prices were all integer
    beta = copy(start_beta)

    # Initialize loose_index, best_obj, and a counter for consecutive improvements
    loose_index = 1

    best_beta = copy(beta)
    consecutive_close = 0

    # Define a threshold for convergence
    conv_thresh = 1e-6
    iter = 0 # failsafe to not run forever in case of non-convergence

    _, _, sLL = compute_sLL_latent5classes(x, y, av, epsilon, R, class_draws, beta, prob_inds, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks)

    best_obj = sLL
    old_obj = sLL
    obj_val = best_obj
    old_beta = copy(best_beta)

    K_here = K + 4 # plus number of class parameter estimates
    
    while consecutive_close < K_here && iter < 5000
        iter += 1        
        result = @timed loose_beta, obj_val = do_BHAMSLE_onedim_latent5classes(x, y, av, epsilon, class_draws, R, beta, loose_index, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, prob_inds, [a[loose_index]], [b[loose_index]], bp_proximity) 

        if loose_index in prob_inds && loose_beta <= 1e-12
            nothing # dont update beta if there were no breakpoints found for probabilites (can actually happen)
        else
            beta[loose_index] = loose_beta
        end
        
        # Check if the current obj_val and beta is very close to the previous one
        if abs(obj_val - old_obj) < conv_thresh
            consecutive_close += 1 # Increment the counter
        else
            beta_close = true
            for k in 1:K_here
                if abs(old_beta[k] - beta[k]) > beta_acc
#                     println("index $k. old_beta[$k] = $(old_beta[k]) != $(beta[k]) = beta[$k], so FALSE")
                    beta_close = false
                else
#                     println("index $k. old_beta[$k] = $(old_beta[k]) == $(beta[k]) = beta[$k], so true")
                end
            end
            if beta_close
                consecutive_close += 1
            else
                consecutive_close = 0  # Reset consecutive_close counter
            end
        end

        if obj_val > best_obj
            best_obj = obj_val
            best_beta = copy(beta)
        end 

        old_obj = copy(obj_val)
        old_beta = copy(beta)

        # Increment loose_index, wrapping around if necessary
        loose_index += 1
        if loose_index > K_here
            loose_index = 1
        end
        
    end
    
    return best_obj, best_beta, iter
end;

function compute_biog_loglike_latent3classes(x, y, av, beta_orig, class_1_ks, class_2_ks, class_3_ks; biogeme=false)
    J = size(x, 1)
    N = size(x, 2)
    K = size(x, 3)
    C = 3
    Vin = zeros(J, N, C)
    Pin = zeros(J, N, C)
    
    beta = copy(beta_orig) # necessary to not change the outside scope beta with permutations
    
    if biogeme
        # biogeme has alphabetically ordered beta
        if length(beta) == 10 # optima 3
            # order should be the same already
            latent_beta = beta[9:10]
            beta = beta[1:8]
        elseif length(beta) == 5 # NETHER MICHEL
            beta[2], beta[3] = beta[3], beta[2]
            latent_beta = beta[4:5]
            beta = beta[1:3]
        elseif length(beta) == 7 && class_1_ks == [1, 2, 3] # NETHER TOM
            # The permutation applied to the array: [1, 3, 5, 2, 4, 6, 7]
            # We need to invert this permutation to [1, 2, 3, 4, 5, 6, 7]
            perm = [1, 3, 5, 2, 4, 6, 7]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[6:7]
            beta = beta[1:5]
        elseif length(beta) == 7 && class_1_ks == [1, 2, 3, 4, 5] && class_2_ks == [1, 2, 3, 5] && class_3_ks == [1, 2, 3, 4] # LONDON MICHEL
            # looks like the order is the same! 
            latent_beta = beta[6:7]
            beta = beta[1:5]
        elseif length(beta) == 9 && class_2_ks != [1, 2, 4, 5, 6] # LONDON TOM
            # The permutation applied to the array: [1, 2, 3, 4, 7, 5, 6, 8, 9]
            # We need to invert this permutation to [1, 2, 3, 4, 5, 6, 7, 8, 9]
            perm = [1, 2, 3, 4, 7, 5, 6, 8, 9]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[8:9]
            beta = beta[1:7]   
        elseif length(beta) == 7 && class_1_ks == [1, 2, 3, 4, 5] && class_2_ks == [1, 2, 4, 5] && class_3_ks == [1, 2, 3, 5] # SM MICHEL
            # we : 1 2 3 4 5
            # biogeme: 1 2 4 5 3
            perm = [1, 2, 4, 5, 3, 6, 7]
            sorted_beta = similar(beta)
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            
            latent_beta = beta[6:7]
            beta = beta[1:5]
        elseif length(beta) == 9 && class_2_ks == [1, 2, 4, 5, 6] # SM TOM
            perm = [1, 2, 4, 7, 5, 3, 6, 8, 9]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[8:9]
            beta = beta[1:7]  
        end
        denom = exp(latent_beta[1]) + exp(latent_beta[2]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = exp(latent_beta[2]) / denom
        P3 = 1 / denom
    else        
        if length(beta) == 10 # optima 3
            latent_beta = beta[9:10]
            beta = beta[1:8]
        elseif length(beta) == 5
            latent_beta = beta[4:5]
            beta = beta[1:3]
        elseif length(beta) == 7
            latent_beta = beta[6:7]
            beta = beta[1:5]
        elseif length(beta) == 9 
            latent_beta = beta[8:9]
            beta = beta[1:7]
        end
        P1 = latent_beta[1]
        P2 = latent_beta[2] - latent_beta[1]
        P3 = 1 - latent_beta[2]
    end
    
    prob = [P1, P2, P3]
    
    for n in 1:N 
        av_n = findall(av[:, n] .== 1)
        for i in av_n
           for c in 1:C
                if c == 1
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                elseif c == 2
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                else
                    Vin[i, n, 3] = sum(beta[k] * x[i, n, k] for k in class_3_ks)
                end
            end
        end
    end

    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for c in 1:C
            # Find the maximum Vin value for the current n and c (log-sum-exp trick)
            max_vin = maximum(Vin[i, n, c] for i in av_n)

            # Compute the denominator using the log-sum-exp trick
            denom = sum(exp(Vin[j, n, c] - max_vin) for j in av_n)

            # Compute Pin, applying the same shift
            for i in av_n
                Pin[i, n, c] = exp(Vin[i, n, c] - max_vin) / denom
            end
        end
    end

    # Compute the overall log-likelihood    
    # biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for i in findall(av[:, n] .== 1) for n in 1:N)
	biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end;

function compute_LL_latent3classes(x, y, av, beta, class_1_ks, class_2_ks, class_3_ks; biogeme=false)
    J = size(x, 1)
    N = size(x, 2)
    K = size(x, 3)
    C = 3
    Vin = zeros(J, N, C)
    Pin = zeros(J, N, C)

    latent_beta = beta[end-1:end]
    beta = beta[1:end-2]
    
    if biogeme
        denom = exp(latent_beta[1]) + exp(latent_beta[2]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = exp(latent_beta[2]) / denom
        P3 = 1 / denom
    else        
        P1 = latent_beta[1]
        P2 = latent_beta[2] - latent_beta[1]
        P3 = 1 - latent_beta[2]
    end
    
    prob = [P1, P2, P3]
    
    for n in 1:N 
        av_n = findall(av[:, n] .== 1)
        for i in av_n
           for c in 1:C
                if c == 1
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                elseif c == 2
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                else
                    Vin[i, n, 3] = sum(beta[k] * x[i, n, k] for k in class_3_ks)
                end
            end
        end
    end

    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for c in 1:C
            # Find the maximum Vin value for the current n and c (log-sum-exp trick)
            max_vin = maximum(Vin[i, n, c] for i in av_n)

            # Compute the denominator using the log-sum-exp trick
            denom = sum(exp(Vin[j, n, c] - max_vin) for j in av_n)

            # Compute Pin, applying the same shift
            for i in av_n
                Pin[i, n, c] = exp(Vin[i, n, c] - max_vin) / denom
            end
        end
    end

    # Compute the overall log-likelihood    
    # biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for i in findall(av[:, n] .== 1) for n in 1:N)
	# biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) <= 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))
    biog_obj = sum(y[i, n] * (sum(Pin[i, n, c] * prob[c] for c in 1:C) <= 0 ? -100 : log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end;

function compute_biog_loglike_latent2classes(x, y, av, beta_orig, class_1_ks, class_2_ks; biogeme=false)
    J = size(x, 1)
    N = size(x, 2)
    K = size(x, 3)
    C = 2
    Vin = zeros(J, N, C)
    Pin = zeros(J, N, C)
    
    beta = copy(beta_orig) # necessary to not change the outside scope beta with permutations
    
    if biogeme
        if length(beta) == 9 # optima 2 and 22
            # order should be the same already
            latent_beta = beta[9]
            beta = beta[1:8]
        elseif length(beta) == 7
            # biogeme has alphabetically ordered beta
            latent_beta = beta[7]
            beta = beta[1:6]
        else # SM has beta length 6
            perm = [1, 2, 4, 5, 3, 6] # this is the starting perm. It makes sense.
            sorted_beta = similar(beta)
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta

            latent_beta = beta[6]
            beta = beta[1:5]
        end
        denom = exp(latent_beta[1]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = 1 / denom
    else       
        if class_2_ks == [1, 2, 3, 4, 6] #LPMC
	        latent_beta = beta[7]
	        beta = beta[1:6]
		elseif class_2_ks == [1, 2, 4, 5] #SM
	        latent_beta = beta[6]
	        beta = beta[1:5]
		end
        if length(beta) == 9 # optima 2 and 22
            # order should be the same already
            latent_beta = beta[9]
            beta = beta[1:8]
        end
        P1 = latent_beta[1]
        P2 = 1 - latent_beta[1]
    end
    
    prob = [P1, P2]
    
    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for i in av_n
           for c in 1:C
                if c == 1
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                else
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                end
            end
        end
    end

    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for c in 1:C
            # Find the maximum Vin value for the current n and c (log-sum-exp trick)
            max_vin = maximum(Vin[i, n, c] for i in av_n)

            # Compute the denominator using the log-sum-exp trick
            denom = sum(exp(Vin[j, n, c] - max_vin) for j in av_n)

            # Compute Pin, applying the same shift
            for i in av_n
                Pin[i, n, c] = exp(Vin[i, n, c] - max_vin) / denom
            end
        end
    end

    # Compute the overall log-likelihood    
    # biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for i in findall(av[:, n] .== 1) for n in 1:N)
	
	# biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) <= 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))	
	biog_obj = sum(y[i, n] * (sum(Pin[i, n, c] * prob[c] for c in 1:C) <= 0 ? -100 : log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end;

function compute_LL_latent2classes(x, y, av, beta, class_1_ks, class_2_ks; biogeme=false)
    J = size(x, 1)
    N = size(x, 2)
    K = size(x, 3)
    C = 2
    Vin = zeros(J, N, C)
    Pin = zeros(J, N, C)

    latent_beta = beta[end]
    beta = beta[1:end-1]
    
    if biogeme
        denom = exp(latent_beta[1]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = 1 / denom
    else       
        P1 = latent_beta[1]
        P2 = 1 - latent_beta[1]
    end
    
    prob = [P1, P2]
    
    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for i in av_n
           for c in 1:C
                if c == 1
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                else
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                end
            end
        end
    end

    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for c in 1:C
            # Find the maximum Vin value for the current n and c (log-sum-exp trick)
            max_vin = maximum(Vin[i, n, c] for i in av_n)

            # Compute the denominator using the log-sum-exp trick
            denom = sum(exp(Vin[j, n, c] - max_vin) for j in av_n)

            # Compute Pin, applying the same shift
            for i in av_n
                Pin[i, n, c] = exp(Vin[i, n, c] - max_vin) / denom
            end
        end
    end

    # Compute the overall log-likelihood    
    # biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for i in findall(av[:, n] .== 1) for n in 1:N)
    # probsum = sum(Pin[i, n, c] * prob[c] for c in 1:C)
	# biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) <= 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))	
	
    # probsum = sum(Pin[i, n, c] * prob[c] for c in 1:C)
    biog_obj = sum(y[i, n] * (sum(Pin[i, n, c] * prob[c] for c in 1:C) <= 0 ? -100 : log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end;

function compute_biog_loglike_latent4classes(x, y, av, beta_orig, class_1_ks, class_2_ks, class_3_ks, class_4_ks; biogeme=false)
    J = size(x, 1)
    N = size(x, 2)
    K = size(x, 3)
    C = 4
    Vin = zeros(J, N, C)
    Pin = zeros(J, N, C)
    
    beta = copy(beta_orig) # necessary to not change the outside scope beta with permutations
    
    if biogeme
        # biogeme has alphabetically ordered beta
        if length(beta) == 8 && class_1_ks == [1, 2, 3, 4, 5] && class_2_ks == [1, 2, 3, 5] && class_3_ks == [1, 2, 3, 4]# LONDON MICHEL			
            # looks like the order is the same! 
            latent_beta = beta[6:8]
            beta = beta[1:5]
        elseif length(beta) == 10 && class_2_ks != [1, 2, 4, 5, 6] # LONDON TOM
            # The permutation applied to the array: [1, 2, 3, 4, 7, 5, 6, 8, 9]
            # We need to invert this permutation to [1, 2, 3, 4, 5, 6, 7, 8, 9]
            perm = [1, 2, 3, 4, 7, 5, 6, 8, 9, 10]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[8:10]
            beta = beta[1:7]   
        elseif length(beta) == 8 && class_1_ks == [1, 2, 3, 4, 5] && class_3_ks == [1, 2, 3, 5] && class_4_ks == [1, 2, 3, 4] # SM MICHEL
            # we : 1 2 3 4 5
            # biogeme: 1 2 4 5 3
            perm = [1, 2, 4, 5, 3, 6, 7, 8]
            sorted_beta = similar(beta)
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            
            latent_beta = beta[6:8]
            beta = beta[1:5]
        elseif length(beta) == 10 && class_2_ks == [1, 2, 4, 5, 6] # SM TOM
            perm = [1, 2, 4, 7, 5, 3, 6, 8, 9, 10]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[8:10]
            beta = beta[1:7]  
        end
        denom = exp(latent_beta[1]) + exp(latent_beta[2]) + exp(latent_beta[3]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = exp(latent_beta[2]) / denom
        P3 = exp(latent_beta[3]) / denom
        P4 = 1 / denom
    else        
        if length(beta) == 6
            latent_beta = beta[4:6]
            beta = beta[1:3]
        elseif length(beta) == 8
            latent_beta = beta[6:8]
            beta = beta[1:5]
        elseif length(beta) == 10 
            latent_beta = beta[8:10]
            beta = beta[1:7]
        end
        P1 = latent_beta[1]
        P2 = latent_beta[2] - latent_beta[1]
        P3 = latent_beta[3] - latent_beta[2]
        P4 = 1 - latent_beta[3]
    end
    
    prob = [P1, P2, P3, P4]
    
    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for i in av_n
           for c in 1:C
                if c == 1
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                elseif c == 2
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                elseif c == 3
                    Vin[i, n, 3] = sum(beta[k] * x[i, n, k] for k in class_3_ks)
                else
                    Vin[i, n, 4] = sum(beta[k] * x[i, n, k] for k in class_4_ks)
                end
            end
        end
    end

    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for c in 1:C
            # Find the maximum Vin value for the current n and c (log-sum-exp trick)
            max_vin = maximum(Vin[i, n, c] for i in av_n)

            # Compute the denominator using the log-sum-exp trick
            denom = sum(exp(Vin[j, n, c] - max_vin) for j in av_n)

            # Compute Pin, applying the same shift
            for i in av_n
                Pin[i, n, c] = exp(Vin[i, n, c] - max_vin) / denom
            end
        end
    end

    # Compute the overall log-likelihood    
	biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end;

function compute_biog_loglike_latent5classes(x, y, av, beta_orig, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks; biogeme=false)
    J = size(x, 1)
    N = size(x, 2)
    K = size(x, 3)
    C = 5
    Vin = zeros(J, N, C)
    Pin = zeros(J, N, C)
    
    beta = copy(beta_orig) # necessary to not change the outside scope beta with permutations
    
    if biogeme
        # biogeme has alphabetically ordered beta
        if length(beta) == 8 && class_1_ks == [1, 2, 3, 4, 5] && class_2_ks == [1, 2, 3, 5] && class_3_ks == [1, 2, 3, 4]# LONDON MICHEL			
            # looks like the order is the same! 
            latent_beta = beta[6:8]
            beta = beta[1:5]
        elseif length(beta) == 10 && class_2_ks != [1, 2, 4, 5, 6] # LONDON TOM
            # The permutation applied to the array: [1, 2, 3, 4, 7, 5, 6, 8, 9]
            # We need to invert this permutation to [1, 2, 3, 4, 5, 6, 7, 8, 9]
            perm = [1, 2, 3, 4, 7, 5, 6, 8, 9, 10]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[8:10]
            beta = beta[1:7]   
        elseif length(beta) == 9 && class_1_ks == [1, 2, 3, 4, 5] && class_3_ks == [1, 2, 3, 5] && class_4_ks == [1, 2, 3, 4] && class_5_ks == [1, 2] # SM MICHEL
            # we : 1 2 3 4 5
            # biogeme: 1 2 4 5 3
            perm = [1, 2, 4, 5, 3, 6, 7, 8, 9]
            sorted_beta = similar(beta)
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            
            latent_beta = beta[6:9]
            beta = beta[1:5]
        elseif length(beta) == 10 && class_2_ks == [1, 2, 4, 5, 6] # SM TOM
            perm = [1, 2, 4, 7, 5, 3, 6, 8, 9, 10]
            # Create an empty array for the sorted values
            sorted_beta = similar(beta)
            # Apply the inverse permutation
            for i in 1:length(perm)
                sorted_beta[perm[i]] = beta[i]
            end
            beta = sorted_beta
            latent_beta = beta[8:10]
            beta = beta[1:7]  
        end
        denom = exp(latent_beta[1]) + exp(latent_beta[2]) + exp(latent_beta[3]) + exp(latent_beta[4]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = exp(latent_beta[2]) / denom
        P3 = exp(latent_beta[3]) / denom
        P4 = exp(latent_beta[4]) / denom
        P5 = 1 / denom
    else        
        if length(beta) == 6
            latent_beta = beta[4:6]
            beta = beta[1:3]
        elseif length(beta) == 9
            latent_beta = beta[6:9]
            beta = beta[1:5]
        elseif length(beta) == 10 
            latent_beta = beta[8:10]
            beta = beta[1:7]
        end
        P1 = latent_beta[1]
        P2 = latent_beta[2] - latent_beta[1]
        P3 = latent_beta[3] - latent_beta[2]
        P4 = latent_beta[4] - latent_beta[3]
        P5 = 1 - latent_beta[4]
    end
    
    prob = [P1, P2, P3, P4, P5]
    
    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for i in av_n
           for c in 1:C
                if c == 1
                    Vin[i, n, 1] = sum(beta[k] * x[i, n, k] for k in class_1_ks)
                elseif c == 2
                    Vin[i, n, 2] = sum(beta[k] * x[i, n, k] for k in class_2_ks)
                elseif c == 3
                    Vin[i, n, 3] = sum(beta[k] * x[i, n, k] for k in class_3_ks)
                elseif c == 4
                    Vin[i, n, 4] = sum(beta[k] * x[i, n, k] for k in class_4_ks)
                else
                    Vin[i, n, 5] = sum(beta[k] * x[i, n, k] for k in class_5_ks)
                end
            end
        end
    end

    for n in 1:N
        av_n = findall(av[:, n] .== 1)
        for c in 1:C
            # Find the maximum Vin value for the current n and c (log-sum-exp trick)
            max_vin = maximum(Vin[i, n, c] for i in av_n)

            # Compute the denominator using the log-sum-exp trick
            denom = sum(exp(Vin[j, n, c] - max_vin) for j in av_n)

            # Compute Pin, applying the same shift
            for i in av_n
                Pin[i, n, c] = exp(Vin[i, n, c] - max_vin) / denom
            end
        end
    end

    # Compute the overall log-likelihood    
	biog_obj = sum(y[i, n] * ifelse(sum(Pin[i, n, c] * prob[c] for c in 1:C) == 0, -100, log(sum(Pin[i, n, c] * prob[c] for c in 1:C))) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end;

function compute_biog_loglike_mixed(x, y, av, beta, mix_inds; R=10000)
    J, N, K = size(x)
    Pin = zeros(J, N)  # Store choice probabilities

    # Extract mean and std for mixed beta coefficients
    distr_params = [k[1] for k in mix_inds]
    means = [beta[k] for k in distr_params]
    stds = [beta[k[2]] for k in mix_inds]

    for n in 1:N
        av_n = findall(av[:, n] .== 1)  # Indices of available alternatives
        R_draws = randn(length(means), R)  # Precompute R samples for efficiency
        beta_samples = means .+ stds .* R_draws  # Generate R samples for random coefficients

        # Compute utilities for all available alternatives and all simulation draws
        Vin_samples = [sum(beta[k] * x[i, n, k] for k in 1:K) + 
                       sum(beta_samples[d, :] .* x[i, n, distr_params[d]] for d in 1:length(distr_params)) 
                       for i in av_n]

        # Compute log-sum-exp for numerical stability across simulation draws
        log_denom = logsumexp(hcat(Vin_samples...), dims=1)  # hcat aligns draws for alternatives

        # Calculate the probabilities for each alternative
        integral_sum = exp.(hcat(Vin_samples...) .- log_denom)  # Numerator minus log-demon
        mean_integral_sum = mean(integral_sum, dims=2)  # Average across R draws
        Pin[av_n, n] .= mean_integral_sum  # Store probabilities for available alternatives
    end

    # Compute the log-likelihood
    biog_obj = sum(y[i, n] * log(Pin[i, n]) for n in 1:N for i in findall(av[:, n] .== 1))

    return biog_obj
end


function run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, starting_point)
    # Define the base directory on the supercomputer
    # base_dir = "/home/thaering/BHAMSLE"
	# base_dir = "/Users/tomhaering/Desktop/JED"
	base_dir = "/home/thaering/BHAMSLE"
    
    # Construct the command for Python with or without the starting point
    if !isnothing(starting_point)
        # Convert starting_point array to a string format suitable for the Python command
        starting_point_str = "'[" * join(starting_point, ", ") * "]'"
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 3 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $OptimaSeed $starting_point_str"
    else
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 3 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $OptimaSeed"
    end

    # Use `bash -c` to execute the command string
    cmd = `bash -c $cmd_str`
    
    # Capture the output of the command
	output = read(cmd, String)

	# Split the output by lines
	lines = split(output, "\n")
	vector_and_time_str = lines[1]

	# Remove the parentheses around the tuple
	cleaned_output = replace(vector_and_time_str, ['(', ')'] => "")

	# Split by the vector, runtime, and result_LL using the closing bracket of the vector as a delimiter
	vector_str, remaining_str = split(cleaned_output, "], ", limit=2)

	# Clean up the vector string and parse the elements
	vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
	result_vector = parse.(Float64, vector_elements)

	# Split the remaining string by the comma to separate runtime and result_LL
	time_str, ll_str = split(remaining_str, ", ", limit=2)

	# Parse the runtime
	result_time = parse(Float64, time_str)

	# Parse the result_LL
	result_LL = parse(Float64, ll_str)
    
    # Compute probabilities
    latent_beta = result_vector[end-1:end]
    denom = exp(latent_beta[1]) + exp(latent_beta[2]) + 1  # exp(0) because one is fixed (like ASC)
    P1 = exp(latent_beta[1]) / denom
    P2 = exp(latent_beta[2]) / denom
    P3 = 1 / denom
    probs = [P1, P2, P3]
    
    return result_vector, result_time, result_LL, probs
end;

function run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, starting_point)
    # Define the base directory on the supercomputer
    base_dir = "/home/thaering/BHAMSLE"
	# base_dir = "/Users/tomhaering/Desktop/JED"
    
    # Construct the command for Python with or without the starting point
    if !isnothing(starting_point)
        # Convert starting_point array to a string format suitable for the Python command
        starting_point_str = "'[" * join(starting_point, ", ") * "]'"
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 2 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $OptimaSeed $starting_point_str"
    else
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 2 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $OptimaSeed"
    end
	
	# python3 execute_instance.py 100 100 2 1 0 0 0 0 0 1

    # Use `bash -c` to execute the command string
    cmd = `bash -c $cmd_str`
    
    # # Split by the vector and the runtime using the closing bracket of the vector as a delimiter
    # vector_str, time_str = split(cleaned_output, "], ")
    #
    # # Clean up the vector string and parse the elements
    # vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
    # result_vector = parse.(Float64, vector_elements)
    #
    # # Parse the runtime
    # result_time = parse(Float64, time_str)
	
	# Capture the output of the command
	output = read(cmd, String)

	# Split the output by lines
	lines = split(output, "\n")
	vector_and_time_str = lines[1]

	# Remove the parentheses around the tuple
	cleaned_output = replace(vector_and_time_str, ['(', ')'] => "")

	# Split by the vector, runtime, and result_LL using the closing bracket of the vector as a delimiter
	vector_str, remaining_str = split(cleaned_output, "], ", limit=2)

	# Clean up the vector string and parse the elements
	vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
	result_vector = parse.(Float64, vector_elements)

	# Split the remaining string by the comma to separate runtime and result_LL
	time_str, ll_str = split(remaining_str, ", ", limit=2)

	# Parse the runtime
	result_time = parse(Float64, time_str)

	# Parse the result_LL
	result_LL = parse(Float64, ll_str)
    
    # Compute probabilities
    latent_beta = result_vector[end:end]
    denom = exp(latent_beta[1]) + 1  # exp(0) because one is fixed (like ASC)
    P1 = exp(latent_beta[1]) / denom
    P2 = 1 / denom
    probs = [P1, P2]
    
    return result_vector, result_time, result_LL, probs
end;

function run_Biogeme_Latent(N, R, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, starting_point)
    # Define the base directory on the supercomputer
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    
    # Construct the command for Python with or without the starting point
    if !isnothing(starting_point)
        # Convert starting_point array to a string format suitable for the Python command
        starting_point_str = "'[" * join(starting_point, ", ") * "]'"
        cmd_str = "python3 $base_dir/execute_instance.py $N $R $C $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $OptimaSeed $starting_point_str"
    else
        cmd_str = "python3 $base_dir/execute_instance.py $N $R $C $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $OptimaSeed"
    end
	
	# python3 execute_instance.py 100 100 2 1 0 0 0 0 0 1

    # Use `bash -c` to execute the command string
    cmd = `bash -c $cmd_str`
    
    # # Split by the vector and the runtime using the closing bracket of the vector as a delimiter
    # vector_str, time_str = split(cleaned_output, "], ")
    #
    # # Clean up the vector string and parse the elements
    # vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
    # result_vector = parse.(Float64, vector_elements)
    #
    # # Parse the runtime
    # result_time = parse(Float64, time_str)
	
	# Capture the output of the command
	# output = read(cmd, String)
	
	output = try
	    read(cmd, String)
	catch e
	    ""
	end

	# Split the output by lines
	lines = split(output, "\n")
	vector_and_time_str = lines[1]

	# Remove the parentheses around the tuple
	cleaned_output = replace(vector_and_time_str, ['(', ')'] => "")

	# Split by the vector, runtime, and result_LL using the closing bracket of the vector as a delimiter
	vector_str, remaining_str = split(cleaned_output, "], ", limit=2)

	# Clean up the vector string and parse the elements
	vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
	result_vector = parse.(Float64, vector_elements)

	# Split the remaining string by the comma to separate runtime and result_LL
	time_str, ll_str = split(remaining_str, ", ", limit=2)

	# Parse the runtime
	result_time = parse(Float64, time_str)

	# Parse the result_LL
	result_LL = parse(Float64, ll_str)
    
    # Compute probabilities
    if string(C)[1] == '2' || string(C)[3] == '2'
        latent_beta = result_vector[end:end]
        denom = exp(latent_beta[1]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = 1 / denom
        probs = [P1, P2]
    elseif string(C)[1] == '3'  || string(C)[3] == '3'
        latent_beta = result_vector[end-1:end]
        denom = exp(latent_beta[1]) + exp(latent_beta[2]) + 1  # exp(0) because one is fixed (like ASC)
        P1 = exp(latent_beta[1]) / denom
        P2 = exp(latent_beta[2]) / denom
        P3 = 1 / denom
        probs = [P1, P2, P3]
    end
    
    return result_vector, result_time, result_LL, probs
end;

function run_Biogeme_mixed(N, R, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starting_point)
    # Define the base directory on the supercomputer
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    
    # Construct the command for Python with or without the starting point
    if !isnothing(starting_point)
        # Convert starting_point array to a string format suitable for the Python command
        starting_point_str = "'[" * join(starting_point, ", ") * "]'"
        cmd_str = "python3 $base_dir/execute_instance.py $N $R $C $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $TseedSM $starting_point_str"
    else
        cmd_str = "python3 $base_dir/execute_instance.py $N $R $C $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $TseedSM"
    end
	
	# python3 execute_instance.py 100 100 2 1 0 0 0 0 0 1

    # Use `bash -c` to execute the command string
    cmd = `bash -c $cmd_str`
    
    # # Split by the vector and the runtime using the closing bracket of the vector as a delimiter
    # vector_str, time_str = split(cleaned_output, "], ")
    #
    # # Clean up the vector string and parse the elements
    # vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
    # result_vector = parse.(Float64, vector_elements)
    #
    # # Parse the runtime
    # result_time = parse(Float64, time_str)
	
	# Capture the output of the command
	output = read(cmd, String)

	# Split the output by lines
	lines = split(output, "\n")
	vector_and_time_str = lines[1]

	# Remove the parentheses around the tuple
	cleaned_output = replace(vector_and_time_str, ['(', ')'] => "")

	# Split by the vector, runtime, and result_LL using the closing bracket of the vector as a delimiter
	vector_str, remaining_str = split(cleaned_output, "], ", limit=2)

	# Clean up the vector string and parse the elements
	vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
	result_vector = parse.(Float64, vector_elements)

	# Split the remaining string by the comma to separate runtime and result_LL
	time_str, ll_str = split(remaining_str, ", ", limit=2)

	# Parse the runtime
	result_time = parse(Float64, time_str)

	# Parse the result_LL
	result_LL = parse(Float64, ll_str)

    return result_vector, result_time, result_LL
end;

function run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starting_point)
    # Define the base directory on the supercomputer
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    
    # Construct the command for Python with or without the starting point
    if !isnothing(starting_point)
        # Convert starting_point array to a string format suitable for the Python command
        starting_point_str = "'[" * join(starting_point, ", ") * "]'"
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 4 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $TseedSM $starting_point_str"
    else
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 4 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $TseedSM"
    end

    # Use `bash -c` to execute the command string
    cmd = `bash -c $cmd_str`
    
    # Capture the output of the command
	output = read(cmd, String)

	# Split the output by lines
	lines = split(output, "\n")
	vector_and_time_str = lines[1]

	# Remove the parentheses around the tuple
	cleaned_output = replace(vector_and_time_str, ['(', ')'] => "")

	# Split by the vector, runtime, and result_LL using the closing bracket of the vector as a delimiter
	vector_str, remaining_str = split(cleaned_output, "], ", limit=2)

	# Clean up the vector string and parse the elements
	vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
	result_vector = parse.(Float64, vector_elements)

	# Split the remaining string by the comma to separate runtime and result_LL
	time_str, ll_str = split(remaining_str, ", ", limit=2)

	# Parse the runtime
	result_time = parse(Float64, time_str)

	# Parse the result_LL
	result_LL = parse(Float64, ll_str)
    
    # Compute probabilities
    latent_beta = result_vector[end-2:end]
    denom = exp(latent_beta[1]) + exp(latent_beta[2]) + exp(latent_beta[3]) + 1  # exp(0) because one is fixed (like ASC)
    P1 = exp(latent_beta[1]) / denom
    P2 = exp(latent_beta[2]) / denom
    P3 = exp(latent_beta[3]) / denom
    P4 = 1 / denom
    probs = [P1, P2, P3, P4]
    
    return result_vector, result_time, result_LL, probs
end;

function run_Biogeme_Nether_Latent_Five_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starting_point)
    # Define the base directory on the supercomputer
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    
    # Construct the command for Python with or without the starting point
    if !isnothing(starting_point)
        # Convert starting_point array to a string format suitable for the Python command
        starting_point_str = "'[" * join(starting_point, ", ") * "]'"
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 5 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $TseedSM $starting_point_str"
    else
        cmd_str = "python3 $base_dir/execute_instance.py $N $R 5 $pandaSeed $Mseed $Tseed $MseedL $TseedL $MseedSM $TseedSM"
    end

    # Use `bash -c` to execute the command string
    cmd = `bash -c $cmd_str`
    
    # Capture the output of the command
	output = read(cmd, String)

	# Split the output by lines
	lines = split(output, "\n")
	vector_and_time_str = lines[1]

	# Remove the parentheses around the tuple
	cleaned_output = replace(vector_and_time_str, ['(', ')'] => "")

	# Split by the vector, runtime, and result_LL using the closing bracket of the vector as a delimiter
	vector_str, remaining_str = split(cleaned_output, "], ", limit=2)

	# Clean up the vector string and parse the elements
	vector_elements = split(replace(vector_str, ['[', ']'] => ""), ", ")
	result_vector = parse.(Float64, vector_elements)

	# Split the remaining string by the comma to separate runtime and result_LL
	time_str, ll_str = split(remaining_str, ", ", limit=2)

	# Parse the runtime
	result_time = parse(Float64, time_str)

	# Parse the result_LL
	result_LL = parse(Float64, ll_str)
    
    # Compute probabilities
    latent_beta = result_vector[end-3:end]
    denom = exp(latent_beta[1]) + exp(latent_beta[2]) + exp(latent_beta[3]) + exp(latent_beta[4]) + 1  # exp(0) because one is fixed (like ASC)
    P1 = exp(latent_beta[1]) / denom
    P2 = exp(latent_beta[2]) / denom
    P3 = exp(latent_beta[3]) / denom
    P4 = exp(latent_beta[4]) / denom
    P5 = 1 / denom
    probs = [P1, P2, P3, P4, P5]
    
    return result_vector, result_time, result_LL, probs
end;

function run_BHAMSLE_Nether_Latent_Three_Classes(N, R, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed; starthelp=false)
    if Mseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_nether_michels_classes_$(Mseed).npz")
        class_1_ks = [1, 2, 3] # = 1:K
        class_2_ks = [1, 2]
        class_3_ks = [1, 3]
        prob_inds = [4, 5] 
    elseif Tseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_nether_toms_extremists_$(Tseed).npz")
        class_1_ks = [1, 2, 3] # = 1:K
        class_2_ks = [1, 3, 4]
        class_3_ks = [1, 2, 5]
        prob_inds = [6, 7] 
    elseif MseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 3, 5] # only care about time
        class_3_ks = [1, 2, 3, 4] # only care about cost
        prob_inds = [6, 7] 
    elseif TseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Tom_$(TseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] # = 1:K
        class_2_ks = [1, 2, 3, 4, 6]
        class_3_ks = [1, 2, 3, 5, 7]
        prob_inds = [8, 9] 
    elseif MseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 4, 5] # dont care about time (3)
        class_3_ks = [1, 2, 3, 5] # dont care about cost (4)
        prob_inds = [6, 7] 
    elseif OptimaSeed > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(OptimaSeed).npz")
	    class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
	    class_2_ks = [1, 2, 3, 4, 5] # ignores all travel and waiting times
	    class_3_ks = [1, 2, 5, 6, 7, 8] # ignores all costs
	    prob_inds = [9, 10]
    else
        error("Neither Mseed or Tseed or MseedL or TseedL or MSeedSM or TSeedSM are > 0")
    end
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
    epsilon = data["epsilon"]; # J N R
    class_draws = data["class_draws"] # N R

    J, N, R = size(epsilon)

    if MseedSM > 0
        old_av = data["av"] # J N, example av[i, n] = 0 or 1
    else 
        old_av = ones(Int, J, N)  # JxN array of integer 1s
    end

    K = size(old_x)[3]
    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    class_draws = reshape(class_draws, N * R) # now sigma is NR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = prob_inds[2] - 2  # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), [0, 0])  # Fill LB repeat_count times and append 0, 0
    b = vcat(fill(UB, repeat_count), [1, 1]);  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, repeat_count), [0.33333, 0.66666]) 
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_latent3classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, prob_inds, a, b, 1e-3)
    end).time
#     starthelp = false
    if !starthelp 
        probs = [round(best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[2]]-best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[2]], digits=3)]
        _, _, CsLL = compute_sLL_latent3classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks, class_3_ks)
        bLL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
        BbLL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
#         println("time_heur     = ",time_heur)
#         println("iterations    = ",iter)
#         println("best_sLL      = ",best_sLL)
#         println("best_betas    = ",best_beta)
#         println("Giving probs  = ",probs)
#         println("Computed sLL  = ",CsLL)
#         println("biogeme LL    = ",bLL)
#         println("Biogemes bLL  = ", BbLL)
#         println("So we are $(better)% better")
        
#         return best_beta, time_heur, old_x, old_y, class_1_ks, class_2_ks, class_3_ks
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        probs = [round(best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[2]]-best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[2]], digits=3)]
        _, _, CsLL = compute_sLL_latent3classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks, class_3_ks)
        bLL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
        BbLL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
        # println("time_heur     = ",time_heur)
        # println("iterations    = ",iter)
        # println("best_sLL      = ",best_sLL)
        # println("best_betas    = ",best_beta)
        # println("Giving probs  = ",probs)
        # println("Computed sLL  = ",CsLL)
        # println("biogeme LL    = ",bLL)
        # println("Biogemes bLL  = ", BbLL)
        # println("So we are $(better)% better")
        
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks
    end
end;

function run_BHAMSLE_mixed(N, R, C, D, mix_inds, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM; starthelp=false)
    if N == 5
        if D == 1 
            data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_nether.npz")
        end
        if D == 2 
            data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_SM.npz")
        end
        if D == 3
            data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_optima.npz")
        end
        if D == 4 
            data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_lpmc.npz")
        end
        if D == 5 
            data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_telephone.npz")
        end
    else
        if D == 1 
            data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_nether.npz")
        end
        if D == 2 
            data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_SM.npz")
        end
        if D == 3 
            data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_optima.npz")
        end
        if D == 4 
            data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_lpmc.npz")
        end
        if D == 5 
            data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_telephone.npz")
        end
    end
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
	
	J, N, K = size(old_x)
	H = length(mix_inds)
	
	epsilon = rand(Gumbel(0, 1), J, N, R)
	sigma = rand(Normal(0, 1), H, N, R)

    old_av = data["av"] # J N, example av[i, n] = 0 or 1

    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    sigma = reshape(sigma, H, N * R) # now sigma is HxNR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = K + H  # I think they all get a bound. (Determine how many times to repeat LB / UB)

    a = fill(LB, repeat_count)  # Fill LB repeat_count times and append 0, 0
    b = fill(UB, repeat_count)  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, K), fill(1, H)) # starting value for STDV params is 1 as in Biogeme
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_mixed(x, y, av, epsilon, sigma, R, start_beta, mix_inds, a, b, 1e-3)
    end).time
	
	# println("")
	# println("BHAMSLE took $iter iterations")
	# println("")
	
#     starthelp = false
    if !starthelp 
        _, _, CsLL = compute_sLL_mixed(x, y, av, epsilon, R, sigma, beta, mix_inds)
        bLL = compute_biog_loglike_mixed(old_x, old_y, old_av, best_beta, mix_inds)
        BbLL = compute_biog_loglike_mixed(old_x, old_y, old_av, biog_beta, mix_inds)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
        # println("time_heur     = ",time_heur)
        # println("iterations    = ",iter)
        # println("best_sLL      = ",best_sLL)
        # println("best_betas    = ",best_beta)
        # println("Giving probs  = ",probs)
        # println("Computed sLL  = ",CsLL)
        # println("biogeme LL    = ",bLL)
        # println("Biogemes bLL  = ", BbLL)
        # println("So we are $(better)% better")
        #
#         return best_beta, time_heur, old_x, old_y, class_1_ks, class_2_ks, class_3_ks
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        # _, _, CsLL = compute_sLL_mixed(x, y, av, epsilon, R, sigma, beta, mix_inds)
        # bLL = compute_biog_loglike_mixed(old_x, old_y, old_av, best_beta, mix_inds)
        
        # println("time_heur     = ",time_heur)
        # println("iterations    = ",iter)
        # println("best_sLL      = ",best_sLL)
        # println("best_betas    = ",best_beta)
        # println("Giving probs  = ",probs)
        # println("Computed sLL  = ",CsLL)
        # println("biogeme LL    = ",bLL)
		
        probs = nothing
        CsLL = nothing
        bLL = nothing
        BbLL = nothing
        better = nothing
		
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av
    end
end;

function run_BHAMSLE_Nether_Latent_Two_Classes(N, R, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed; starthelp=false)
    if MseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 3, 4, 6] 
        prob_inds = [7] 
    end
	
    if MseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
    	class_2_ks = [1, 2, 4, 5] # dont care about time (3)
    	prob_inds = [6] 
    end

    if OptimaSeed > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(OptimaSeed).npz")
	    class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
	    class_2_ks = [1, 2, 3, 4, 5] # ignores all travel and waiting times
	    prob_inds = [9]
    end

    if TseedL > 0 # hidden second optima two class option
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(TseedL).npz")
	    class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
	    class_2_ks = [1, 2, 3, 4, 5, 7, 8] # ignores only CAR travel time
	    prob_inds = [9]
    end
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
    epsilon = data["epsilon"]; # J N R
    class_draws = data["class_draws"] # N R

    J, N, R = size(epsilon)

    if MseedSM > 0 
        old_av = data["av"] # J N, example av[i, n] = 0 or 1
    else 
        old_av = ones(Int, J, N)  # JxN array of integer 1s
    end

    K = size(old_x)[3]
    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    class_draws = reshape(class_draws, N * R) # now sigma is NR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = prob_inds[1] - 1  # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), [0])  # Fill LB repeat_count times and append 0, 0
    b = vcat(fill(UB, repeat_count), [1]);  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, repeat_count), [0.5]) 
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_latent2classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, prob_inds, a, b, 1e-3)
    end).time
#     starthelp = false
    if !starthelp 
        probs = [round(best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[1]], digits=3)]
        _, _, CsLL = compute_sLL_latent2classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks)
        bLL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, biogeme=false)
        BbLL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
        # println("time_heur     = ",time_heur)
        # println("iterations    = ",iter)
        # println("best_sLL      = ",best_sLL)
        # println("best_betas    = ",best_beta)
        # println("Giving probs  = ",probs)
        # println("Computed sLL  = ",CsLL)
        # println("biogeme LL    = ",bLL)
        # println("Biogemes bLL  = ", BbLL)
        # println("So we are $(better)% better")
        #
#         return best_beta, time_heur, old_x, old_y, class_1_ks, class_2_ks, class_3_ks
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        probs = [round(best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[1]], digits=3)]
        _, _, CsLL = compute_sLL_latent2classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks)
        bLL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, biogeme=false)
        
        # println("time_heur     = ",time_heur)
        # println("iterations    = ",iter)
        # println("best_sLL      = ",best_sLL)
        # println("best_betas    = ",best_beta)
        # println("Giving probs  = ",probs)
        # println("Computed sLL  = ",CsLL)
        # println("biogeme LL    = ",bLL)
		
        probs = nothing
        CsLL = nothing
        bLL = nothing
        BbLL = nothing
        better = nothing
		
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av, class_1_ks, class_2_ks
    end
end;

function run_BHAMSLE_Latent_Two_Classes(N, R, C, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed; starthelp=false)
    if Mseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_nether_Michel_$(MseedL).npz")
        # 1 ASC rail
        # 2 Cost
        # 3 Time
        if C == 2
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2] 
            prob_inds = [4] 
        elseif C == 22
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 3] 
            prob_inds = [4] 
        elseif C == 23
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1] 
            prob_inds = [4] 
        end
    end

    if MseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz")
        # ASC_CAR      -0.185907      0.482518    -0.385285      0.700026
        # ASC_TRAIN    -0.330812      0.616437    -0.536651      0.591509
        # B_COST       -1.839381      0.683981    -2.689229      0.007162
        # B_HE         -3.878408      8.934273    -0.434104      0.664213
        # B_TIME  
        if C == 22 
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 5] 
            prob_inds = [6]
        elseif C == 23
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            prob_inds = [6]
        elseif C == 24
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2] 
            prob_inds = [6]
        end
    end

    if OptimaSeed > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(OptimaSeed).npz")
        # ASC_CAR             0.289219      0.712165     0.406113      0.684660
        # ASC_SM             -0.155131      0.980680    -0.158187      0.874310
        # BETA_COST_HWH     -25.632326     16.309223    -1.571646      0.116033
        # BETA_COST_OTHER    -5.226712      4.526484    -1.154696      0.248215
        # BETA_DIST          -2.925026      2.265154    -1.291314      0.196595
        # BETA_TIME_CAR     -21.029007     24.024233    -0.875325      0.381397
        # BETA_TIME_PT       -8.713693     10.444978    -0.834247      0.404142
        # BETA_WAITING_TIME  -0.074130      0.149418    -0.496126      0.619806
	    if C == 22
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 5, 6, 7, 8]
            prob_inds = [9]
        elseif C == 23
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2]
            prob_inds = [9]
        end
    end
    
    if MseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz")
        # ASC_Bike      -4.195427      0.777725    -5.394489  6.871879e-08
        # ASC_Car       -1.700855      0.609906    -2.788715  5.291762e-03
        # ASC_PB        -0.688697      0.397275    -1.733551  8.299782e-02
        # Beta_cost     -0.165182      0.072914    -2.265444  2.348545e-02
        # Beta_time     -6.807629      1.512680    -4.500375  6.783375e-06
        if C == 21
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4] 
            prob_inds = [6] 
        elseif C == 22
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            prob_inds = [6] 
        elseif C == 23
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3] 
            prob_inds = [6] 
        end
    end
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
    epsilon = data["epsilon"]; # J N R
    class_draws = data["class_draws"] # N R

    J, N, R = size(epsilon)

    if MseedSM > 0 
        old_av = data["av"] # J N, example av[i, n] = 0 or 1
    else 
        old_av = ones(Int, J, N)  # JxN array of integer 1s
    end

    K = size(old_x)[3]
    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    class_draws = reshape(class_draws, N * R) # now sigma is NR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = prob_inds[1] - 1  # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), [0])  # Fill LB repeat_count times and append 0, 0
    b = vcat(fill(UB, repeat_count), [1]);  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, repeat_count), [0.5]) 
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_latent2classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, prob_inds, a, b, 1e-3)
    end).time
#     starthelp = false
    if !starthelp 
        probs = [round(best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[1]], digits=3)]
        _, _, CsLL = compute_sLL_latent2classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks)
        bLL = compute_LL_latent2classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, biogeme=false)
        BbLL = compute_LL_latent2classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        probs = [round(best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[1]], digits=3)]
        _, _, CsLL = compute_sLL_latent2classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks)
        bLL = compute_LL_latent2classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, biogeme=false)
        
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av, class_1_ks, class_2_ks
    end
end;

function run_BHAMSLE_Latent_Three_Classes(N, R, C, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed; starthelp=false)
    if Mseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_nether_Michel_$(MseedL).npz")
        # 1 ASC rail
        # 2 Cost
        # 3 Time
        if C == 3
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2] 
            class_3_ks = [1, 3]
            prob_inds = [4, 5] 
        elseif C == 31
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2] 
            class_3_ks = [1]
            prob_inds = [4, 5] 
        elseif C == 32
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 3] 
            class_3_ks = [1]
            prob_inds = [4, 5] 
        end
    end

    if MseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz")
        # ASC_CAR      -0.185907      0.482518    -0.385285      0.700026
        # ASC_TRAIN    -0.330812      0.616437    -0.536651      0.591509
        # B_COST       -1.839381      0.683981    -2.689229      0.007162
        # B_HE         -3.878408      8.934273    -0.434104      0.664213
        # B_TIME  
        if C == 31 
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 5] 
            class_3_ks = [1, 2]
            prob_inds = [6, 7]
        elseif C == 32
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 5] 
            class_3_ks = [1, 2]
            prob_inds = [6, 7]
        elseif C == 33
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            class_3_ks = [1, 2]
            prob_inds = [6, 7]
        end
    end

    if OptimaSeed > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(OptimaSeed).npz")
        # ASC_CAR             0.289219      0.712165     0.406113      0.684660
        # ASC_SM             -0.155131      0.980680    -0.158187      0.874310
        # BETA_COST_HWH     -25.632326     16.309223    -1.571646      0.116033
        # BETA_COST_OTHER    -5.226712      4.526484    -1.154696      0.248215
        # BETA_DIST          -2.925026      2.265154    -1.291314      0.196595
        # BETA_TIME_CAR     -21.029007     24.024233    -0.875325      0.381397
        # BETA_TIME_PT       -8.713693     10.444978    -0.834247      0.404142
        # BETA_WAITING_TIME  -0.074130      0.149418    -0.496126      0.619806
        if C == 31
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 3, 4, 5]
            class_3_ks = [1, 2]
            prob_inds = [9, 10]
	    elseif C == 32
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 5, 6, 7, 8]
            class_3_ks = [1, 2]
            prob_inds = [9, 10]
        end
    end
    
    if MseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz")
        # ASC_Bike      -4.195427      0.777725    -5.394489  6.871879e-08
        # ASC_Car       -1.700855      0.609906    -2.788715  5.291762e-03
        # ASC_PB        -0.688697      0.397275    -1.733551  8.299782e-02
        # Beta_cost     -0.165182      0.072914    -2.265444  2.348545e-02
        # Beta_time     -6.807629      1.512680    -4.500375  6.783375e-06
        if C == 3
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4] 
            class_3_ks = [1, 2, 3, 5]
            prob_inds = [6, 7] 
        elseif C == 31
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4]
            class_3_ks = [1, 2, 3]
            prob_inds = [6, 7] 
        elseif C == 32
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            class_3_ks = [1, 2, 3]
            prob_inds = [6, 7] 
        end
    end
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
    epsilon = data["epsilon"]; # J N R
    class_draws = data["class_draws"] # N R

    J, N, R = size(epsilon)

    if MseedSM > 0
        old_av = data["av"] # J N, example av[i, n] = 0 or 1
    else 
        old_av = ones(Int, J, N)  # JxN array of integer 1s
    end

    K = size(old_x)[3]
    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    class_draws = reshape(class_draws, N * R) # now sigma is NR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = prob_inds[2] - 2  # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), [0, 0])  # Fill LB repeat_count times and append 0, 0
    b = vcat(fill(UB, repeat_count), [1, 1]);  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, repeat_count), [0.33333, 0.66666]) 
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_latent3classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, prob_inds, a, b, 1e-3)
    end).time
#     starthelp = false
    if !starthelp 
        probs = [round(best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[2]]-best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[2]], digits=3)]
        _, _, CsLL = compute_sLL_latent3classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks, class_3_ks)
        bLL = compute_LL_latent3classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
        BbLL = compute_LL_latent3classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
#         println("time_heur     = ",time_heur)
#         println("iterations    = ",iter)
#         println("best_sLL      = ",best_sLL)
#         println("best_betas    = ",best_beta)
#         println("Giving probs  = ",probs)
#         println("Computed sLL  = ",CsLL)
#         println("biogeme LL    = ",bLL)
#         println("Biogemes bLL  = ", BbLL)
#         println("So we are $(better)% better")
        
#         return best_beta, time_heur, old_x, old_y, class_1_ks, class_2_ks, class_3_ks
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        probs = [round(best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[2]]-best_beta[prob_inds[1]], digits=3), round(1-best_beta[prob_inds[2]], digits=3)]
        _, _, CsLL = compute_sLL_latent3classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks, class_3_ks)
        bLL = compute_LL_latent3classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
        BbLL = compute_LL_latent3classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
        # println("time_heur     = ",time_heur)
        # println("iterations    = ",iter)
        # println("best_sLL      = ",best_sLL)
        # println("best_betas    = ",best_beta)
        # println("Giving probs  = ",probs)
        # println("Computed sLL  = ",CsLL)
        # println("biogeme LL    = ",bLL)
        # println("Biogemes bLL  = ", BbLL)
        # println("So we are $(better)% better")
        
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks
    end
end;

function run_BHAMSLE_Latent(N, R, C, D, biog_beta, pandaSeed, class_1_ks, class_2_ks, class_3_ks, extra_inds, prob_inds, mix_inds, class_1_av, class_2_av, class_3_av)
    if 1020 <= C <= 1039
        if N == 5
            if D == 1 
                data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_nether.npz")
            end
            if D == 2 
                data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_SM.npz")
            end
            if D == 3
                data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_optima.npz")
            end
            if D == 4 
                data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_lpmc.npz")
            end
            if D == 5 
                data = npzread("startNPZ/input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_telephone.npz")
            end
        else
            if D == 1 
                data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_nether.npz")
            end
            if D == 2 
                data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_SM.npz")
            end
            if D == 3 
                data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_optima.npz")
            end
            if D == 4 
                data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_lpmc.npz")
            end
            if D == 5 
                data = npzread("input_data_$(N)_mixed_latent_$(pandaSeed)_$(C)_telephone.npz")
            end
        end
    else
        if D == 1
            if N == 5
                data = npzread("startNPZ/input_data_$(N)_$(R)_latent_$(pandaSeed)_nether_michels_classes_1.npz")
            else
                data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_nether_michels_classes_1.npz")
            end
        elseif D == 2
            if N == 5
                data = npzread("startNPZ/input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_1.npz")
            else
                data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_1.npz")
            end
        elseif D == 3
            if N == 5
                data = npzread("startNPZ/nput_data_$(N)_$(R)_latent_$(pandaSeed)_optima_1.npz")
            else
                data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_1.npz")
            end
        elseif D == 4
            if N == 5
                data = npzread("startNPZ/input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_1.npz")
            else
                data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_1.npz")
            end
        elseif D == 5
            if N == 5
                data = npzread("startNPZ/input_data_$(N)_$(R)_latent_$(pandaSeed)_telephone_1.npz")
            else
                data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_telephone_1.npz")
            end
        end
    end
    
    old_x = data["x"] # J N K
	J, N, K = size(old_x)

    # extend x for mixed indices
    for pair in mix_inds
        old_idx = pair[1]  # we just add the needed one at the end. As long as they are ordered, this is fine.
        slice_to_add = old_x[:, :, old_idx]  # Extract the slice
        old_x = cat(old_x, reshape(slice_to_add, size(old_x, 1), size(old_x, 2), 1), dims=3)  # Append along third dimension
    end

    # extend x for latent classes
    for pair in extra_inds
        old_idx = pair[1]  # we just add the needed one at the end. As long as they are ordered, this is fine.
        slice_to_add = old_x[:, :, old_idx]  # Extract the slice
        old_x = cat(old_x, reshape(slice_to_add, size(old_x, 1), size(old_x, 2), 1), dims=3)  # Append along third dimension
    end

    old_y = data["y"] # J N
    old_av = data["av"] # J N 

    H = length(mix_inds)
	
	Random.seed!(1) 
	
	epsilon = rand(Gumbel(0, 1), J, N * R)
	sigma = rand(Normal(0, 1), H, N * R)
    class_draws = rand(Uniform(0, 1), N * R)

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    av = repeat(old_av, 1, R)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
	
	if !isnothing(extra_inds)
		E = length(extra_inds)
	else
		E = 0
	end
	
	# println("size(x) = ", size(x))
	# println("size(sigma) = ", size(sigma))
	# println("J = ", J)
	# println("H = ", H)
	# println("K = ", K)
	# println("E = ", E)
	
	# adjust x for mixed 
	if H >= 1
	    for k in K+1:K+H
	        # x[:, :, k] .= x[:, :, k] .* (1 .+ sigma[k - K, :])'
	        for i in 1:J
	            # Multiply by (1 + sigma)
	            x[i, :, k] .= x[i, :, k] .* (1 .+ sigma[k - K, :])
	        end
	    end
	end
    
    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])
	
	# println("prob_inds = ", prob_inds)

    if string(C)[1] == '2' || string(C)[3] == '2'
        repeat_count = prob_inds[1] - 1 
        actualC = 2
    elseif string(C)[1] == '3' || string(C)[3] == '3'
        repeat_count = prob_inds[2] - 2
        actualC = 3
    end

      # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), fill(0, actualC-1))  # Fill LB repeat_count times and append 0s for prob
    b = vcat(fill(UB, repeat_count), fill(1, actualC-1));  # Fill UB repeat_count times and append 1s for prob
	
    start_beta = vcat(fill(0, K), fill(1, H), fill(0, E), [round(i/actualC,digits=5) for i in 1:(actualC-1)]) 
    # print("start_beta = ", start_beta)
	# println("length = ", length(start_beta))
	
    time_heur = (@timed begin
        if actualC == 2
            best_sLL, best_beta, iter = BHAMSLE_latent2classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, prob_inds, a, b, class_1_av, class_2_av, 1e-3)
        elseif actualC == 3
            best_sLL, best_beta, iter = BHAMSLE_latent3classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, prob_inds, a, b, class_1_av, class_2_av, class_3_av, 1e-3)
        end
    end).time
    if actualC == 2
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av
    elseif actualC == 3
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av
    end
end;

function run_BHAMSLE_Nether_Latent_Four_Classes(N, R, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM; starthelp=false)
    if Mseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_nether_michels_classes_$(Mseed).npz")
        class_1_ks = [1, 2, 3] # = 1:K
        class_2_ks = [1, 2]
        class_3_ks = [1, 3]
        class_4_ks = [1]
        prob_inds = [4, 5, 6] 
    elseif Tseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_nether_toms_extremists_$(Tseed).npz")
        class_1_ks = [1, 2, 3] # = 1:K
        class_2_ks = [1, 3, 4]
        class_3_ks = [1, 2, 5]
        class_4_ks = [1]
        prob_inds = [6, 7, 8] 
    elseif MseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 3, 5] # only care about time
        class_3_ks = [1, 2, 3, 4] # only care about cost
        class_4_ks = [1, 2, 3]
        prob_inds = [6, 7, 8] 
    elseif TseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Tom_$(TseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] # = 1:K
        class_2_ks = [1, 2, 3, 4, 6]
        class_3_ks = [1, 2, 3, 5, 7]
        class_4_ks = [1, 2, 3]
        prob_inds = [8, 9, 10] 
    elseif MseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 4, 5] # dont care about time (3)
        class_3_ks = [1, 2, 3, 5] # dont care about cost (4)
        class_4_ks = [1, 2, 3, 4] # dont care about both, but still headway OR not care about HE
        prob_inds = [6, 7, 8] 
    elseif TseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Tom_$(TseedSM).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 4, 5, 6]
        class_3_ks = [1, 2, 3, 5, 7]
        class_4_ks = [1, 2]
        prob_inds = [8, 9, 10] 
    else
        error("Neither Mseed or Tseed or MseedL or TseedL are > 0")
    end
    
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
    epsilon = data["epsilon"]; # J N R
    class_draws = data["class_draws"] # N R

    J, N, R = size(epsilon)

    if MseedSM > 0 || TseedSM > 0
        old_av = data["av"] # J N, example av[i, n] = 0 or 1
    else 
        old_av = ones(Int, J, N)  # JxN array of integer 1s
    end

    K = size(old_x)[3]
    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    class_draws = reshape(class_draws, N * R) # now sigma is NR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = prob_inds[3] - 3  # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), [0, 0, 0])  # Fill LB repeat_count times and append 0, 0
    b = vcat(fill(UB, repeat_count), [1, 1, 1]);  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, repeat_count), [0.25, 0.5, 0.75]) 
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_latent4classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, prob_inds, a, b, 1e-3)
    end).time
#     starthelp = false
    if !starthelp 
        probs = [round(best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[2]]-best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[3]]-best_beta[prob_inds[2]], digits=3), round(1-best_beta[prob_inds[3]], digits=3)]
        _, _, CsLL = compute_sLL_latent4classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks, class_3_ks, class_4_ks)
        bLL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=false)
        BbLL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
#         println("time_heur     = ",time_heur)
#         println("iterations    = ",iter)
#         println("best_sLL      = ",best_sLL)
#         println("best_betas    = ",best_beta)
#         println("Giving probs  = ",probs)
#         println("Computed sLL  = ",CsLL)
#         println("biogeme LL    = ",bLL)
#         println("Biogemes bLL  = ", BbLL)
#         println("So we are $(better)% better")
        
#         return best_beta, time_heur, old_x, old_y, class_1_ks, class_2_ks, class_3_ks
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        probs = nothing
        CsLL = nothing
        bLL = nothing
        BbLL = nothing
        better = nothing
        
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, class_4_ks
    end
end;

function run_BHAMSLE_Nether_Latent_Five_Classes(N, R, biog_beta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM; starthelp=false)
    if Mseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_nether_michels_classes_$(Mseed).npz")
        class_1_ks = [1, 2, 3] # = 1:K
        class_2_ks = [1, 2]
        class_3_ks = [1, 3]
        class_4_ks = [1]
        prob_inds = [4, 5, 6] 
    elseif Tseed > 0
        data = npzread("input_data_$(N)_$(R)_latent_nether_toms_extremists_$(Tseed).npz")
        class_1_ks = [1, 2, 3] # = 1:K
        class_2_ks = [1, 3, 4]
        class_3_ks = [1, 2, 5]
        class_4_ks = [1]
        prob_inds = [6, 7, 8] 
    elseif MseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 3, 5] # only care about time
        class_3_ks = [1, 2, 3, 4] # only care about cost
        class_4_ks = [1, 2, 3]
        prob_inds = [6, 7, 8] 
    elseif TseedL > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Tom_$(TseedL).npz")
        class_1_ks = [1, 2, 3, 4, 5] # = 1:K
        class_2_ks = [1, 2, 3, 4, 6]
        class_3_ks = [1, 2, 3, 5, 7]
        class_4_ks = [1, 2, 3]
        prob_inds = [8, 9, 10] 
    elseif MseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 4, 5] # dont care about time (3)
        class_3_ks = [1, 2, 3, 5] # dont care about cost (4)
        class_4_ks = [1, 2, 3, 4] # dont care about both, but still headway OR not care about HE
        class_5_ks = [1, 2] # ignore all three
        prob_inds = [6, 7, 8, 9] 
    elseif TseedSM > 0
        data = npzread("input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Tom_$(TseedSM).npz")
        class_1_ks = [1, 2, 3, 4, 5] 
        class_2_ks = [1, 2, 4, 5, 6]
        class_3_ks = [1, 2, 3, 5, 7]
        class_4_ks = [1, 2]
        prob_inds = [8, 9, 10] 
    else
        error("Neither Mseed or Tseed or MseedL or TseedL are > 0")
    end
    
    
    old_x = data["x"] # J N K
    old_y = data["y"] # J N
    epsilon = data["epsilon"]; # J N R
    class_draws = data["class_draws"] # N R

    J, N, R = size(epsilon)

    if MseedSM > 0 || TseedSM > 0
        old_av = data["av"] # J N, example av[i, n] = 0 or 1
    else 
        old_av = ones(Int, J, N)  # JxN array of integer 1s
    end

    K = size(old_x)[3]
    epsilon = reshape(epsilon, J, N * R); # now epsilon is J NR
    class_draws = reshape(class_draws, N * R) # now sigma is NR

    y = repeat(old_y, 1, R)[:, vec(repeat(1:size(old_y, 2), inner=R))] # This repeats each entry of y R times
    x = repeat(old_x, 1, R, 1)[:, vec(repeat(1:size(old_x, 2), inner=R)), :]
    av = repeat(old_av, 1, R, 1)[:, vec(repeat(1:size(old_av, 2), inner=R)), :]

    LB = minimum([- maximum(abs.(biog_beta)) * 1.5, -12])
    UB = maximum([maximum(abs.(biog_beta)) * 1.5, 12])

    repeat_count = prob_inds[4] - 4  # Determine how many times to repeat LB / UB

    a = vcat(fill(LB, repeat_count), [0, 0, 0, 0])  # Fill LB repeat_count times and append 0, 0
    b = vcat(fill(UB, repeat_count), [1, 1, 1, 1]);  # Fill UB repeat_count times and append 1, 1
    
    start_beta = vcat(fill(0, repeat_count), [0.2, 0.4, 0.6, 0.8]) 
    time_heur = (@timed begin
    best_sLL, best_beta, iter = BHAMSLE_latent5classes(x, y, av, epsilon, class_draws, R, start_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, prob_inds, a, b, 1e-3)
    end).time
#     starthelp = false
    if !starthelp 
        probs = [round(best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[2]]-best_beta[prob_inds[1]], digits=3), round(best_beta[prob_inds[3]]-best_beta[prob_inds[2]], digits=3), round(best_beta[prob_inds[4]]-best_beta[prob_inds[3]], digits=3), round(1-best_beta[prob_inds[4]], digits=3)]
        _, _, CsLL = compute_sLL_latent5classes(x, y, av, epsilon, R, class_draws, best_beta, prob_inds, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks)
        bLL = compute_biog_loglike_latent5classes(old_x, old_y, old_av, best_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, biogeme=false)
        BbLL = compute_biog_loglike_latent5classes(old_x, old_y, old_av, biog_beta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, biogeme=true)
        better = round(((bLL - BbLL) / bLL) * 100, digits=2) 
        
#         println("time_heur     = ",time_heur)
#         println("iterations    = ",iter)
#         println("best_sLL      = ",best_sLL)
#         println("best_betas    = ",best_beta)
#         println("Giving probs  = ",probs)
#         println("Computed sLL  = ",CsLL)
#         println("biogeme LL    = ",bLL)
#         println("Biogemes bLL  = ", BbLL)
#         println("So we are $(better)% better")
        
#         return best_beta, time_heur, old_x, old_y, class_1_ks, class_2_ks, class_3_ks
        
        return better, time_heur, best_sLL, CsLL, bLL, BbLL, best_beta, probs
    else
        probs = nothing
        CsLL = nothing
        bLL = nothing
        BbLL = nothing
        better = nothing
        
        return best_sLL, best_beta, time_heur, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks
    end
end;

# Warm-up function to ensure compilation
function warm_up_3_LPMC()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    bhamsleBeta, _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Three_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_2()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    bhamsleBeta, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_2_SM()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)	
    _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_mixed_SM(C)
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _ = run_Biogeme_mixed(N, R, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _ = run_BHAMSLE_mixed(N, R, C, 2, mix_inds, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_mixed(C, D, mix_inds)
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1

    if D == 1 # nether
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 1, 0, 0, 0, 0, 0
    elseif D == 2 # SM
		pandaSeed = 13
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0
    elseif D == 3 # optima
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1
    elseif D == 4 # lmc
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0
    elseif D == 5 # telephone
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0
    end

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _ = run_Biogeme_mixed(N, R, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _ = run_BHAMSLE_mixed(N, R, C, D, mix_inds, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_2_optima()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed = 0, 0, 0, 0, 0, 1

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, nothing)	
    _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, starthelp=true)
end

function warm_up_22_optima()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed = 0, 0, 0, 1, 0, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, nothing)	
    _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, OptimaSeed, starthelp=true)
end

# Warm-up function to ensure compilation
function warm_up_4()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    bhamsleBeta, _, _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Four_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_3_SM()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Three_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_3_optima()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Three_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_4_SM()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Four_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function warm_up_5_SM()
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1
    Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

    # Run Biogeme and BHAMSLE with small values
    FirstbioBeta, _, _, _ = run_Biogeme_Nether_Latent_Five_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _, _, _, _, _ = run_BHAMSLE_Nether_Latent_Five_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)
end

function main_3_SM(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_3_SM()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(7)
	total_bhamsle_beta = zeros(7)
	total_bio_SecondBeta = zeros(7)
	total_bio_FirstProbs = zeros(3)
	total_bhamsle_probs = zeros(3)
	total_bio_SecondProbs = zeros(3)
	
	prob_inds = [6, 7] 

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks = run_BHAMSLE_Nether_Latent_Three_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
		
		# bhamsle 
		
		# k = 1: ASC_Car
		# k = 2: ASC_Train
		# k = 3: Beta_time
		# k = 4: Beta_cost
		# k = 5: Beta_headway
		
		# alphabetical 
		
		# k = 1: ASC_Car
		# k = 2: ASC_Train
		# k = 4: Beta_cost
		# k = 5: Beta_headway
		# k = 3: Beta_time
		
        perm = [1, 2, 5, 3, 4, 6, 7] # this is the starting perm of US when viewed as Bio
        sorted_beta = similar(bhamsleBeta)
        for i in 1:length(perm)
            sorted_beta[perm[i]] = bhamsleBeta[i]
        end
        bio_start_beta = sorted_beta

        x, y = bio_start_beta[end-1:end]
        P1, P2, P3 = x, y - x, 1 - y
        bio_start_beta[end-1:end] .= [round(log(P1) - log(P3), digits=3), round(log(P2) - log(P3), digits=3)]


        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz`)

        # Track results

        # well this needs some adjustement

        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta

		total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3), round(FirstbioProbs[3], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[2]]-bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[2]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3), round(SecondbioProbs[3], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_3_SM.json", results)
end

function main_3_optima(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_3_optima()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(10)
	total_bhamsle_beta = zeros(10)
	total_bio_SecondBeta = zeros(10)
	total_bio_FirstProbs = zeros(3)
	total_bhamsle_probs = zeros(3)
	total_bio_SecondProbs = zeros(3)
	
	prob_inds = [9, 10] 

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks = run_BHAMSLE_Nether_Latent_Three_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
        bio_start_beta = copy(bhamsleBeta)

        x, y = bio_start_beta[end-1:end]
        P1, P2, P3 = x, y - x, 1 - y
        bio_start_beta[end-1:end] .= [round(log(P1) - log(P3), digits=3), round(log(P2) - log(P3), digits=3)]


        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(TseedSM).npz`)

        # Track results

        # well this needs some adjustement

        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta

		total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3), round(FirstbioProbs[3], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[2]]-bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[2]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3), round(SecondbioProbs[3], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_3_optima.json", results)
end

function main_3_LPMC(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_3_LPMC()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_improvemente = 0.0

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, _ = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks = run_BHAMSLE_Nether_Latent_Three_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
		
		# BHAMLSE BETA ORDER
        # 1 ASC_Bike
        # 2 ASC_Car
        # 3 ASC_PT
        # 4 Beta_cost
        # 5 Beta_time
		# 6 PR1
		# 7 PR2
		
		# BIO BETA ORDER
		
		# ASC_Bike     -507.267922     38.420324   -13.203114  0.000000e+00
		# ASC_Car       -10.120993      1.836423    -5.511255  3.562844e-08
		# ASC_PB       -301.604857     22.654939   -13.312985  0.000000e+00
		# Beta_cost    -377.416129     28.753510   -13.125915  0.000000e+00
		# Beta_time    -309.155372     23.979588   -12.892439  0.000000e+00
		# prob_class_1   16.265855      1.000395    16.259432  0.000000e+00
		# prob_class_2 
		
		# might just be the same!
		
        bio_start_beta = copy(bhamsleBeta)
        x, y = bio_start_beta[end-1:end]
        P1, P2, P3 = x, y - x, 1 - y
        bio_start_beta[end-1:end] .= [round(log(P1) - log(P3), digits=3), round(log(P2) - log(P3), digits=3)]

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, _ = run_Biogeme_Nether_Latent_Three_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
        Second_Bio_LL = compute_biog_loglike_latent3classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Tom_$(TseedL).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
        total_improvemente += ((First_Bio_LL - Second_Bio_LL) / First_Bio_LL) * 100
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
    avg_improvemente = total_improvemente / (seed_end - seed_start + 1)
    real_improvemente = ((avg_LL_1 - avg_LL_2) / avg_LL_1) * 100

    # Save results to a file
    results = [N, R, avg_LL_1, avg_bioTime_1, avg_LL_2, avg_bhamsleTime, avg_bioTime_2, real_improvemente, avg_improvemente]
    writedlm("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_3.csv", results, ',')
end

function main_2(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_2()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(7)
	total_bhamsle_beta = zeros(7)
	total_bio_SecondBeta = zeros(7)
	total_bio_FirstTimeFactor = 0.0
	total_bhamsle_timeFactor = 0.0
	total_bio_SecondTimeFactor = 0.0
	total_bio_FirstProbs = zeros(2)
	total_bhamsle_probs = zeros(2)
	total_bio_SecondProbs = zeros(2)
	
	prob_inds = [7]

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
		
		# BHAMLSE BETA ORDER
        # 1 ASC_Bike
        # 2 ASC_Car
        # 3 ASC_PT
        # 4 Beta_cost
        # 5 Beta_time
		# 6 Beta_time_C2
		# 7 PR1
		
		# BIO BETA ORDER
		
		# ASC_Bike     -507.267922     38.420324   -13.203114  0.000000e+00
		# ASC_Car       -10.120993      1.836423    -5.511255  3.562844e-08
		# ASC_PB       -301.604857     22.654939   -13.312985  0.000000e+00
		# Beta_cost    -377.416129     28.753510   -13.125915  0.000000e+00
		# Beta_time    -309.155372     23.979588   -12.892439  0.000000e+00
		# prob_class_1   16.265855      1.000395    16.259432  0.000000e+00
		# prob_class_2 
		
		# might just be the same!
		
        bio_start_beta = copy(bhamsleBeta)
        x = bio_start_beta[end]
        P1, P2 = x, 1 - x
        bio_start_beta[end] = round(log(P1) - log(P2), digits=3)

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, biogeme=true)
		println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
        Second_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, biogeme=true)
		# println("SecondbioLL = ", SecondbioLL)
		# println("Second_Bio_LL = ", Second_Bio_LL)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta
		# 5 is time, 6 is time_c2
		total_bio_FirstTimeFactor += FirstbioBeta[5] / FirstbioBeta[6]
		total_bhamsle_timeFactor += bhamsleBeta[5] / bhamsleBeta[6]
		total_bio_SecondTimeFactor += SecondbioBeta[5] / SecondbioBeta[6]
		total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[1]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstTimeFactor = total_bio_FirstTimeFactor / (seed_end - seed_start + 1)
	avg_bhamsle_timeFactor = total_bhamsle_timeFactor / (seed_end - seed_start + 1)
	avg_bio_SecondTimeFactor = total_bio_SecondTimeFactor / (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)
	
	# Replace NaN values with a placeholder, e.g., "undefined"
	if isnan(avg_bio_FirstTimeFactor)
	    avg_bio_FirstTimeFactor = 1
	end

	if isnan(avg_bhamsle_timeFactor)
	    avg_bhamsle_timeFactor = 1
	end

	if isnan(avg_bio_SecondTimeFactor)
	    avg_bio_SecondTimeFactor = 1
	end

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstTimeFactor" => avg_bio_FirstTimeFactor,
	    "avg_bhamsle_timeFactor" => avg_bhamsle_timeFactor,
	    "avg_bio_SecondTimeFactor" => avg_bio_SecondTimeFactor,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_2.json", results)
	
	
	# results = JSON3.read("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_2.json")
	
	

    # # Save results to a file
    # results = [N, R, avg_LL_1, avg_bioTime_1, avg_LL_2, avg_bhamsleTime, avg_bioTime_2, avg_bio_FirstBeta, avg_bhamsle_beta, avg_bio_SecondBeta, avg_bio_FirstTimeFactor, avg_bhamsle_timeFactor, avg_bio_SecondTimeFactor, avg_bio_FirstProbs, avg_bhamsle_probs, avg_bio_SecondProbs]
    # writedlm("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_2.csv", results, ',')
end

function main_2_SM(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_2_SM()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(6)
	total_bhamsle_beta = zeros(6)
	total_bio_SecondBeta = zeros(6)
	total_bio_FirstProbs = zeros(2)
	total_bhamsle_probs = zeros(2)
	total_bio_SecondProbs = zeros(2)
	
	prob_inds = [6]

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2
		
		# BHAMLSE BETA ORDER
        # k = 1: ASC_Car
        # k = 2: ASC_Train
        # k = 3: Beta_time
        # k = 4: Beta_cost
        # k = 5: Beta_headway
        # k = 6: Prob1
		
		# BIO BETA ORDER
		
		# ASC_CAR       0.026497      0.597072     0.044379      0.964602
        # ASC_TRAIN    -0.288124      0.649597    -0.443543      0.657373
        # B_COST       -2.163711      0.868060    -2.492582      0.012682
        # B_HE         -1.610553     10.144885    -0.158755      0.873862
        # B_TIME       -3.376539      1.508765    -2.237949      0.025224
        # prob_class_1  1.749947      0.944815     1.852157      0.064003

        perm = [1, 2, 5, 3, 4, 6] # starting permutation as seen from Bio perspective
        sorted_beta = similar(bhamsleBeta)
        for i in 1:length(perm)
            sorted_beta[perm[i]] = bhamsleBeta[i]
        end
        bio_start_beta = sorted_beta

        x = bio_start_beta[end]
        P1, P2 = x, 1 - x
        bio_start_beta[end] = round(log(P1) - log(P2), digits=3)

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)

		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta
		
        total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[1]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_2_SM.json", results)
end

function simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, beta, C, R=1000)
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    python_script = joinpath(base_dir, "simulate_ML_SM.py")

    beta_str = "[" * join(beta, ", ") * "]"

    cmd_str = "python3 $python_script $N $pandaSeed \"$beta_str\" $C $R"
    cmd = `bash -c $cmd_str`

    output = read(cmd, String)
    logLikelihood = parse(Float64, strip(output))

    return logLikelihood
end

function simulate_likelihood_mixed_optima_julia(N, pandaSeed, beta, C, R=1000)
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    python_script = joinpath(base_dir, "simulate_ML_optima.py")

    beta_str = "[" * join(beta, ", ") * "]"

    cmd_str = "python3 $python_script $N $pandaSeed \"$beta_str\" $C $R"
    cmd = `bash -c $cmd_str`

    output = read(cmd, String)
    logLikelihood = parse(Float64, strip(output))

    return logLikelihood
end

function simulate_likelihood_mixed_lpmc_julia(N, pandaSeed, beta, C, R=1000)
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    python_script = joinpath(base_dir, "simulate_ML_lpmc.py")

    beta_str = "[" * join(beta, ", ") * "]"

    cmd_str = "python3 $python_script $N $pandaSeed \"$beta_str\" $C $R"
    cmd = `bash -c $cmd_str`

    output = read(cmd, String)
    logLikelihood = parse(Float64, strip(output))

    return logLikelihood
end

function simulate_likelihood_mixed_telephone_julia(N, pandaSeed, beta, C, R=1000)
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    python_script = joinpath(base_dir, "simulate_ML_telephone.py")

    beta_str = "[" * join(beta, ", ") * "]"

    cmd_str = "python3 $python_script $N $pandaSeed \"$beta_str\" $C $R"
    cmd = `bash -c $cmd_str`

    output = read(cmd, String)
    logLikelihood = parse(Float64, strip(output))

    return logLikelihood
end

function simulate_likelihood_mixed(N, D, pandaSeed, beta, C, R=1000)
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    if D == 1
        python_script = joinpath(base_dir, "simulate_ML_nether.py")
    elseif D == 2
        python_script = joinpath(base_dir, "simulate_ML_SM.py")
    elseif D == 3
        python_script = joinpath(base_dir, "simulate_ML_optima.py")
    elseif D == 4
        python_script = joinpath(base_dir, "simulate_ML_lpmc.py")
    elseif D == 5
        python_script = joinpath(base_dir, "simulate_ML_telephone.py")
    end

    beta_str = "[" * join(beta, ", ") * "]"

    cmd_str = "python3 $python_script $N $pandaSeed \"$beta_str\" $C $R"
	
	# println("cmd_str = ", cmd_str)
	
    cmd = `bash -c $cmd_str`

    output = read(cmd, String)
	
	# println("output = ", output)
	
    logLikelihood = parse(Float64, strip(output))

    return logLikelihood
end

function simulate_likelihood_mixed_nether_julia(N, pandaSeed, beta, C, R=1000)
    # base_dir = "/home/thaering/BHAMSLE"
	base_dir = @__DIR__
	# base_dir = "/Users/tomhaering/Desktop/JED"
    python_script = joinpath(base_dir, "simulate_ML_nether.py")

    beta_str = "[" * join(beta, ", ") * "]"

    cmd_str = "python3 $python_script $N $pandaSeed \"$beta_str\" $C $R"
    cmd = `bash -c $cmd_str`

    output = read(cmd, String)
    logLikelihood = parse(Float64, strip(output))

    return logLikelihood
end


function compute_bLL_mixed(x, y, av, beta_orig, mix_inds, R=100)
    """
    Computes the log-likelihood for a mixed logit model using Monte Carlo simulation.

    Parameters:
    - x: Input data, 3D array of size (J, N, K).
    - y: Observed choices, binary matrix of size (J, N).
    - av: Availability matrix, binary matrix of size (J, N).
    - beta_orig: Vector of estimated parameters (mean and standard deviations).
    - mix_inds: Vector of tuples, where each tuple contains the index of a parameter and its standard deviation.
    - R: Number of Monte Carlo draws.

    Returns:
    - Log-likelihood value for the mixed logit model.
    """
    # Set random seed
    Random.seed!(192)

    # Adjust indices for Julia's 1-based indexing
    mix_inds = [(index[1], index[2]) for index in mix_inds]

    J, N, K = size(x)
    num_mixed = length(mix_inds)
    num_means = length(beta_orig) - num_mixed

    beta_mean = beta_orig[1:num_means]
    beta_std = zeros(num_means)

    for (mean_idx, std_idx) in mix_inds
        beta_std[mean_idx] = beta_orig[std_idx]
    end

    # Monte Carlo draws for random effects
    random_effects = randn(R, num_means)

    # Compute utility for each draw
    Vin = zeros(J, N, R)
    for r in 1:R
        beta_r = beta_mean .+ random_effects[r, :] .* beta_std

        for n in 1:N
            for i in 1:J
                if av[i, n] == 1
                    Vin[i, n, r] = sum(beta_r[k] * x[i, n, k] for k in 1:num_means)
                end
            end
        end
    end

    # Compute probabilities for each draw
    Pin = zeros(J, N, R)
    for r in 1:R
        for n in 1:N
            available_indices = findall(av[:, n] .== 1)
            max_vin = maximum(Vin[i, n, r] for i in available_indices)
            denom = sum(exp(Vin[i, n, r] - max_vin) for i in available_indices)
            for i in available_indices
                Pin[i, n, r] = exp(Vin[i, n, r] - max_vin) / denom
            end
        end
    end

    # Average probabilities across all draws
    P_avg = mean(Pin, dims=3)[:, :, 1]

    # Compute the overall log-likelihood
    biog_obj = 0.0
    for n in 1:N
        for i in 1:J
            if av[i, n] == 1
                log_prob = P_avg[i, n] == 0 ? -100.0 : log(P_avg[i, n])
                biog_obj += y[i, n] * log_prob
            end
        end
    end

    return biog_obj
end;


function main_mixed_SM(N, R, C, seed_start, seed_end, highR)
    
    if C == 10
        mix_inds = [[5, 6]]  # mix just time
    elseif C == 11
        mix_inds = [[5, 6], [3, 7]]  # mix time and costs
    elseif C == 12
        mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
    end
    
    # Run warm-up to ensure the code is compiled
    warm_up_mixed_SM(C, mix_inds)

    # 1 ASC_CAR      -1.050295      1.504237    -0.698224      0.485037
    # 2 ASC_TRAIN     0.469923      0.865527     0.542933      0.587176
    # 3 B_COST       -1.926358      1.064709    -1.809281      0.070407
    # 4 B_HE        -55.498268     60.538947    -0.916737      0.359281
    # 5 B_TIME       -3.109267      1.875963    -1.657424      0.097434
    # 6 Z1_B_TIME_S  -3.646606      3.321097    -1.098013      0.272199
    # 7 Z2_B_COST_S  -1.050104      0.726699    -1.445033      0.148449
    # 8 Z3_B_HE_S    39.408984     37.913371     1.039448      0.298596

    betalen = 5

    if C == 10
        mix_inds = [[5, 6]]  # mix just time
        betalen += 1
    elseif C == 11
        mix_inds = [[5, 6], [3, 7]]  # mix time and costs
        betalen += 2
    elseif C == 12
        mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
        betalen += 3
    else
        mix_inds = nothing
    end

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_LL_1_sim = 0.0
    total_LL_2_sim = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bhamsle_LL_sim = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(betalen)
	total_bhamsle_beta = zeros(betalen)
	total_bio_SecondBeta = zeros(betalen)

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0
		
		if highR
			bioR = 1000
		else
			bioR = R
		end

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL = run_Biogeme_mixed(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av = run_BHAMSLE_mixed(N, R, C, 2, mix_inds, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        bio_start_beta = copy(bhamsleBeta)

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL = run_Biogeme_mixed(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        LL_R = bioR

        First_Bio_LL = compute_bLL_mixed(old_x, old_y, old_av, FirstbioBeta, mix_inds, LL_R)
        bhamsle_LL = compute_bLL_mixed(old_x, old_y, old_av, bhamsleBeta, mix_inds, LL_R)
        Second_Bio_LL = compute_bLL_mixed(old_x, old_y, old_av, SecondbioBeta, mix_inds, LL_R)

        First_Bio_LL_sim = simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, FirstbioBeta, C, LL_R)
        bhamsle_LL_sim = simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, bhamsleBeta, C, LL_R)
        Second_Bio_LL_sim = simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, SecondbioBeta, C, LL_R)

        
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL_sim = ", First_Bio_LL_sim)
		println("First_Bio_LL = ", First_Bio_LL)
		println("FirstbioBeta = ", FirstbioBeta)
        println("FirstbioTime = ", FirstbioTime)
		
        println("bhamsle_sLL = ", bhamsle_sLL)
		println("bhamsle_LL_sim = ", bhamsle_LL_sim)
		println("bhamsle_LL = ", bhamsle_LL)
		println("bhamsleBeta = ", bhamsleBeta)
        println("bhamsleTime = ", bhamsleTime)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL_sim = ", Second_Bio_LL_sim)
		println("Second_Bio_LL = ", Second_Bio_LL)
		println("SecondbioBeta = ", SecondbioBeta)
        println("SecondbioTime = ", SecondbioTime)

		# remove data generated in the process. 
		# run(`rm input_data_$(N)_mixed_$(pandaSeed)_SM.npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_LL_1_sim += First_Bio_LL_sim
        total_LL_2_sim += Second_Bio_LL_sim
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bhamsle_LL_sim += bhamsle_LL_sim
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_LL_1_sim = total_LL_1_sim / (seed_end - seed_start + 1)
    avg_LL_2_sim = total_LL_2_sim / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bhamsle_LL_sim = total_bhamsle_LL_sim / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
        "avg_LL_1_sim" => avg_LL_1_sim,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_LL_2_sim" => avg_LL_2_sim,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
        "avg_bhamsle_LL_sim" => avg_bhamsle_LL_sim,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_mixed_SM.json", results)
end

function main_mixed(N, R, C, D, seed_start, seed_end, highR)
    if D == 1 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 1, 0, 0, 0, 0, 0
        betalen = 3
	    if C == 10
	    	mix_inds = [[3, 4]]  # mix just time
	    elseif C == 11
	    	mix_inds = [[3, 4], [2, 5]]  # mix time and costs
	    elseif C == 12
	    	mix_inds = [[3, 4], [2, 5], [1, 6]]  # mix time and costs and ASC rail
	    elseif C == 13
	        mix_inds = [[2, 4]]  # mix only costs
	    elseif C == 14
	        mix_inds = [[1, 4]]  # mix ASC rail
		end 
    end
    if D == 2 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0
        betalen = 5
	    if C == 10
            mix_inds = [[5, 6]]  # mix just time
        elseif C == 11
            mix_inds = [[5, 6], [3, 7]]  # mix time and costs
        elseif C == 12
            mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
        end
    end
    if D == 3
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1
        betalen = 8
        if C == 10
            mix_inds = [[6, 9], [7, 10]]  # mix just time
        elseif C == 11
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12]]  # mix time and costs
        elseif C == 12
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13]]  # mix time and costs and distance
        elseif C == 13
            mix_inds = [[6, 9], [7, 10], [5, 11]]  # mix time and distance
        elseif C == 14
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13], [1, 14], [2, 15]]  # mix time and costs and
            # distance and ASCs
        elseif C == 15
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [1, 13], [2, 14]]  # mix time and costs and ASCs
        elseif C == 16
            mix_inds = [[6, 9], [7, 10], [1, 11], [2, 12]]  # mix time and ASCs
        elseif C == 17
            mix_inds = [[1, 9], [2, 10]]  # mix ASCs only
        elseif C == 18
            mix_inds = [[1, 9]]  # mix ASC Car only
        end
    end
    if D == 4 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0
        betalen = 5
	    if C == 10
            mix_inds = [[5, 6]]  # mix just time
        elseif C == 11
            mix_inds = [[5, 6], [4, 7]] # mix time and costs
        elseif C == 12
            mix_inds = [[4, 6]] # mix costs
        elseif C == 18
            mix_inds = [[2, 6]]  # mix ASC Car
        end
    end
    if D == 5 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0
        betalen = 5
        if C == 10
            mix_inds = [[5, 6]]  # mix cost
        elseif C == 11
            mix_inds = [[5, 6], [1, 7], [2, 8], [3, 9], [4, 10]]  # mix cost and ASCs
        elseif C == 12
            mix_inds = [[1, 6], [2, 7], [3, 8], [4, 9]]  # mix ASCs
		end
    end

    betalen += length(mix_inds)

    # Run warm-up to ensure the code is compiled
    warm_up_mixed(C, D, mix_inds)

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_LL_1_sim = 0.0
    total_LL_2_sim = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bhamsle_LL_sim = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(betalen)
	total_bhamsle_beta = zeros(betalen)
	total_bio_SecondBeta = zeros(betalen)

    HR = 0
	bioR = R
	
	# if highR
	#     HR = 1
	# 	bioR = 1000
	# end
	
    for pandaSeed in seed_start:seed_end
        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL = run_Biogeme_mixed(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av = run_BHAMSLE_mixed(N, R, C, D, mix_inds, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        bio_start_beta = copy(bhamsleBeta)

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL = run_Biogeme_mixed(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
		
		LL_R = R
		
		if LL_R < 1000
			LL_R = 1000
		end

        First_Bio_LL = compute_bLL_mixed(old_x, old_y, old_av, FirstbioBeta, mix_inds, LL_R)
        bhamsle_LL = compute_bLL_mixed(old_x, old_y, old_av, bhamsleBeta, mix_inds, LL_R)
        Second_Bio_LL = compute_bLL_mixed(old_x, old_y, old_av, SecondbioBeta, mix_inds, LL_R)
        if D == 1
            First_Bio_LL_sim = simulate_likelihood_mixed_nether_julia(N, pandaSeed, FirstbioBeta, C, LL_R)
            bhamsle_LL_sim = simulate_likelihood_mixed_nether_julia(N, pandaSeed, bhamsleBeta, C, LL_R)
            Second_Bio_LL_sim = simulate_likelihood_mixed_nether_julia(N, pandaSeed, SecondbioBeta, C, LL_R)
        elseif D == 2
            First_Bio_LL_sim = simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, FirstbioBeta, C, LL_R)
            bhamsle_LL_sim = simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, bhamsleBeta, C, LL_R)
            Second_Bio_LL_sim = simulate_likelihood_mixed_swissmetro_julia(N, pandaSeed, SecondbioBeta, C, LL_R)
        elseif D == 3
            First_Bio_LL_sim = simulate_likelihood_mixed_optima_julia(N, pandaSeed, FirstbioBeta, C, LL_R)
            bhamsle_LL_sim = simulate_likelihood_mixed_optima_julia(N, pandaSeed, bhamsleBeta, C, LL_R)
            Second_Bio_LL_sim = simulate_likelihood_mixed_optima_julia(N, pandaSeed, SecondbioBeta, C, LL_R)
        elseif D == 4
            First_Bio_LL_sim = simulate_likelihood_mixed_lpmc_julia(N, pandaSeed, FirstbioBeta, C, LL_R)
            bhamsle_LL_sim = simulate_likelihood_mixed_lpmc_julia(N, pandaSeed, bhamsleBeta, C, LL_R)
            Second_Bio_LL_sim = simulate_likelihood_mixed_lpmc_julia(N, pandaSeed, SecondbioBeta, C, LL_R)
        elseif D == 5
            First_Bio_LL_sim = simulate_likelihood_mixed_telephone_julia(N, pandaSeed, FirstbioBeta, C, LL_R)
            bhamsle_LL_sim = simulate_likelihood_mixed_telephone_julia(N, pandaSeed, bhamsleBeta, C, LL_R)
            Second_Bio_LL_sim = simulate_likelihood_mixed_telephone_julia(N, pandaSeed, SecondbioBeta, C, LL_R)
        end
        
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL_sim = ", First_Bio_LL_sim)
		# println("First_Bio_LL = ", First_Bio_LL)
		println("FirstbioBeta = ", FirstbioBeta)
        println("FirstbioTime = ", FirstbioTime)
		
        println("bhamsle_sLL = ", bhamsle_sLL)
		println("bhamsle_LL_sim = ", bhamsle_LL_sim)
		# println("bhamsle_LL = ", bhamsle_LL)
		println("bhamsleBeta = ", bhamsleBeta)
        println("bhamsleTime = ", bhamsleTime)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL_sim = ", Second_Bio_LL_sim)
		# println("Second_Bio_LL = ", Second_Bio_LL)
		println("SecondbioBeta = ", SecondbioBeta)
        println("SecondbioTime = ", SecondbioTime)

		# remove data generated in the process. 
		# run(`rm input_data_$(N)_mixed_$(pandaSeed)_SM.npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_LL_1_sim += First_Bio_LL_sim
        total_LL_2_sim += Second_Bio_LL_sim
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bhamsle_LL_sim += bhamsle_LL_sim
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_LL_1_sim = total_LL_1_sim / (seed_end - seed_start + 1)
    avg_LL_2_sim = total_LL_2_sim / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bhamsle_LL_sim = total_bhamsle_LL_sim / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
        "avg_LL_1_sim" => avg_LL_1_sim,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_LL_2_sim" => avg_LL_2_sim,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
        "avg_bhamsle_LL_sim" => avg_bhamsle_LL_sim,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta
	)

	if D == 1
        JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_$(HR)_mixed_nether.json", results)
    elseif D == 2
        JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_$(HR)_mixed_SM.json", results)
    elseif D == 3
        JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_$(HR)_mixed_optima.json", results)
    elseif D == 4
        JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_$(HR)_mixed_lpmc.json", results)
    elseif D == 5
        JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_$(HR)_mixed_telephone.json", results)
    end
end


function main_2_optima(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_2_optima()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(9)
	total_bhamsle_beta = zeros(9)
	total_bio_SecondBeta = zeros(9)
	total_bio_FirstProbs = zeros(2)
	total_bhamsle_probs = zeros(2)
	total_bio_SecondProbs = zeros(2)
	
	prob_inds = [9]

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # I think the order of the betas should be the same
        bio_start_beta = copy(bhamsleBeta)

        x = bio_start_beta[end]
        P1, P2 = x, 1 - x
        bio_start_beta[end] = round(log(P1) - log(P2), digits=3)

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)

		# remove data generated in the process. 
        run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(TseedSM).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta
		
        total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[1]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_2_optima.json", results)
end

function main_22_optima(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_22_optima()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(9)
	total_bhamsle_beta = zeros(9)
	total_bio_SecondBeta = zeros(9)
	total_bio_FirstProbs = zeros(2)
	total_bhamsle_probs = zeros(2)
	total_bio_SecondProbs = zeros(2)

    prob_inds = [9]

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks = run_BHAMSLE_Nether_Latent_Two_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2
        bio_start_beta = copy(bhamsleBeta)

        x = bio_start_beta[end]
        P1, P2 = x, 1 - x
        bio_start_beta[end] = round(log(P1) - log(P2), digits=3)

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Two_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent2classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)

		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_optima_$(TseedL).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta
		
        total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[1]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_22_optima.json", results)
end

function main_4_SM(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_4_SM()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(8)
	total_bhamsle_beta = zeros(8)
	total_bio_SecondBeta = zeros(8)
	total_bio_FirstProbs = zeros(4)
	total_bhamsle_probs = zeros(4)
	total_bio_SecondProbs = zeros(4)

    prob_inds = [6, 7, 8]

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, class_4_ks = run_BHAMSLE_Nether_Latent_Four_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
		
		perm = [1, 2, 5, 3, 4, 6, 7, 8] # this is the starting perm of US when viewed as Bio
        sorted_beta = similar(bhamsleBeta)
        for i in 1:length(perm)
            sorted_beta[perm[i]] = bhamsleBeta[i]
        end
        bio_start_beta = sorted_beta

        x, y, z = bio_start_beta[end-2:end]
        P1, P2, P3, P4 = x, y - x, z - y, 1 - z
        bio_start_beta[end-2:end] .= [round(log(P1) - log(P4), digits=3), round(log(P2) - log(P4), digits=3), round(log(P3) - log(P4), digits=3)]

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta

		total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3), round(FirstbioProbs[3], digits=3), round(FirstbioProbs[4], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[2]]-bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[3]]-bhamsleBeta[prob_inds[2]], digits=3), round(1-bhamsleBeta[prob_inds[3]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3), round(SecondbioProbs[3], digits=3), round(SecondbioProbs[4], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_4_SM.json", results)
end

function main_5_SM(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_5_SM()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bhamsle_sLL = 0.0
    total_bhamsle_LL = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_bio_FirstBeta = zeros(9)
	total_bhamsle_beta = zeros(9)
	total_bio_SecondBeta = zeros(9)
	total_bio_FirstProbs = zeros(5)
	total_bhamsle_probs = zeros(5)
	total_bio_SecondProbs = zeros(5)

    prob_inds = [6, 7, 8, 9]

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Nether_Latent_Five_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks = run_BHAMSLE_Nether_Latent_Five_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
		
		perm = [1, 2, 5, 3, 4, 6, 7, 8, 9] # this is the starting perm of US when viewed as Bio
        sorted_beta = similar(bhamsleBeta)
        for i in 1:length(perm)
            sorted_beta[perm[i]] = bhamsleBeta[i]
        end
        bio_start_beta = sorted_beta

        x, y, z, zz = bio_start_beta[end-3:end]
        P1, P2, P3, P4, P5 = x, y - x, z - y, zz - z, 1-zz
        bio_start_beta[end-3:end] .= [round(log(P1) - log(P5), digits=3), round(log(P2) - log(P5), digits=3), round(log(P3) - log(P5), digits=3), round(log(P4) - log(P5), digits=3)]

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Nether_Latent_Five_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent5classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, biogeme=true)
        bhamsle_LL = compute_biog_loglike_latent5classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, biogeme=false)
        Second_Bio_LL = compute_biog_loglike_latent5classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, class_5_ks, biogeme=true)
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
		
		println("bhamsle_LL = ", bhamsle_LL)
		
        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_SM_Michel_$(MseedSM).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bhamsle_sLL += bhamsle_sLL
        total_bhamsle_LL += bhamsle_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
	    total_bio_FirstBeta .+= FirstbioBeta
		total_bhamsle_beta .+= bhamsleBeta
		total_bio_SecondBeta .+= SecondbioBeta

		total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3), round(FirstbioProbs[3], digits=3), round(FirstbioProbs[4], digits=3), round(FirstbioProbs[5], digits=3)]
		total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[2]]-bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[3]]-bhamsleBeta[prob_inds[2]], digits=3), round(bhamsleBeta[prob_inds[4]]-bhamsleBeta[prob_inds[3]], digits=3), round(1-bhamsleBeta[prob_inds[4]], digits=3)]
		total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3), round(SecondbioProbs[3], digits=3), round(SecondbioProbs[4], digits=3), round(SecondbioProbs[5], digits=3)]
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	results = Dict(
	    "N" => N,
	    "R" => R,
	    "avg_LL_1" => avg_LL_1,
	    "avg_bioTime_1" => avg_bioTime_1,
	    "avg_LL_2" => avg_LL_2,
        "avg_bhamsle_sLL" => avg_bhamsle_sLL,
        "avg_bhamsle_LL" => avg_bhamsle_LL,
	    "avg_bhamsleTime" => avg_bhamsleTime,
	    "avg_bioTime_2" => avg_bioTime_2,
	    "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	    "avg_bhamsle_beta" => avg_bhamsle_beta,
	    "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	    "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	    "avg_bhamsle_probs" => avg_bhamsle_probs,
	    "avg_bio_SecondProbs" => avg_bio_SecondProbs
	)

	JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_5_SM.json", results)
end

function main_4(N, R, seed_start, seed_end)
    # Run warm-up to ensure the code is compiled
    warm_up_4()

    # Initialize arrays to store results
    total_LL_1 = 0.0
    total_LL_2 = 0.0
    total_bioTime_1 = 0.0
    total_bhamsleTime = 0.0
    total_bioTime_2 = 0.0
    total_improvemente = 0.0

    for pandaSeed in seed_start:seed_end
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0

        # Run Biogeme and BHAMSLE for the first estimation
        FirstbioBeta, FirstbioTime, _ = run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsleBeta, bhamsleTime, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, class_4_ks = run_BHAMSLE_Nether_Latent_Four_Classes(N, R, FirstbioBeta, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, starthelp=true)

        # Apply permutation and compute probabilities P1, P2, P3
		
		# BHAMLSE BETA ORDER
        # 1 ASC_Bike
        # 2 ASC_Car
        # 3 ASC_PT
        # 4 Beta_cost
        # 5 Beta_time
		# 6 PR1
		# 7 PR2
		
		# BIO BETA ORDER
		
		# ASC_Bike     -507.267922     38.420324   -13.203114  0.000000e+00
		# ASC_Car       -10.120993      1.836423    -5.511255  3.562844e-08
		# ASC_PB       -301.604857     22.654939   -13.312985  0.000000e+00
		# Beta_cost    -377.416129     28.753510   -13.125915  0.000000e+00
		# Beta_time    -309.155372     23.979588   -12.892439  0.000000e+00
		# prob_class_1   16.265855      1.000395    16.259432  0.000000e+00
		# prob_class_2 
		
		# might just be the same!
		
        bio_start_beta = copy(bhamsleBeta)
        x, y, z = bio_start_beta[end-2:end]
        P1, P2, P3, P4 = x, y - x, z - y, 1 - z
        bio_start_beta[end-2:end] .= [round(log(P1) - log(P4), digits=3), round(log(P2) - log(P4), digits=3), round(log(P3) - log(P4), digits=3)]

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, _ = run_Biogeme_Nether_Latent_Four_Classes(N, R, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)

        # Compute log-likelihoods
        First_Bio_LL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=true)
        Second_Bio_LL = compute_biog_loglike_latent4classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, class_4_ks, biogeme=true)
		
		# remove data generated in the process. 
		run(`rm input_data_$(N)_$(R)_latent_$(pandaSeed)_lpmc_Michel_$(MseedL).npz`)

        # Track results
        total_LL_1 += First_Bio_LL
        total_LL_2 += Second_Bio_LL
        total_bioTime_1 += FirstbioTime
        total_bhamsleTime += bhamsleTime
        total_bioTime_2 += SecondbioTime
        total_improvemente += ((First_Bio_LL - Second_Bio_LL) / First_Bio_LL) * 100
    end

    # Compute averages
    avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
    avg_improvemente = total_improvemente / (seed_end - seed_start + 1)
    real_improvemente = ((avg_LL_1 - avg_LL_2) / avg_LL_1) * 100

    # Save results to a file
    results = [N, R, avg_LL_1, avg_bioTime_1, avg_LL_2, avg_bhamsleTime, avg_bioTime_2, real_improvemente, avg_improvemente]
    writedlm("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_4.csv", results, ',')
end

function warm_up_nat_latent(C, D, class_1_ks, class_2_ks, class_3_ks, extra_inds, prob_inds, mix_inds)
    # Use small values to compile functions
    N, R, pandaSeed = 5, 5, 1

    if D == 1 # nether
		J = 2
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 1, 0, 0, 0, 0, 0
    elseif D == 2 # SM
		J = 3
		pandaSeed = 13
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0
    elseif D == 3 # optima
		J = 3
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1
    elseif D == 4 # lmc
		J = 4
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0
    elseif D == 5 # telephone
		J = 5
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0
    end
	
	class_1_av = collect(1:J) 
	class_2_av = collect(1:J) 
	class_3_av = collect(1:J)

    FirstbioBeta, _, _, _ = run_Biogeme_Latent(N, R, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
    _, _, _, _, _, _ = run_BHAMSLE_Latent(N, R, C, D, FirstbioBeta, pandaSeed, class_1_ks, class_2_ks, class_3_ks, extra_inds, prob_inds, mix_inds, class_1_av, class_2_av, class_3_av)
end

function main_nat_latent(N, R, C, D, seed_start, seed_end)
    if D == 1 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 1, 0, 0, 0, 0, 0
        betalen = 3
		J = 2
	    if C == 1010
	    	mix_inds = [[3, 4]]  # mix just time
	    elseif C == 1011
	    	mix_inds = [[3, 4], [2, 5]]  # mix time and costs
	    elseif C == 1012
	    	mix_inds = [[3, 4], [2, 5], [1, 6]]  # mix time and costs and ASC rail
	    elseif C == 1013
	        mix_inds = [[2, 4]]  # mix only costs
	    elseif C == 1014
	        mix_inds = [[1, 4]]  # mix ASC rail
		end 
    end
    if D == 2 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 1, 0
        betalen = 5
		J = 3
	    if C == 1010
            mix_inds = [[5, 6]]  # mix just time
        elseif C == 1011
            mix_inds = [[5, 6], [3, 7]]  # mix time and costs
        elseif C == 1012
            mix_inds = [[5, 6], [3, 7], [4, 8]]  # mix time and costs and headway
        end
    end
    if D == 3
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 0, 0, 1
        betalen = 8
		J = 3
        if C == 1010
            mix_inds = [[6, 9], [7, 10]]  # mix just time
        elseif C == 1011
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12]]  # mix time and costs
        elseif C == 1012
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13]]  # mix time and costs and distance
        elseif C == 1013
            mix_inds = [[6, 9], [7, 10], [5, 11]]  # mix time and distance
        elseif C == 1014
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [5, 13], [1, 14], [2, 15]]  # mix time and costs and
            # distance and ASCs
        elseif C == 1015
            mix_inds = [[6, 9], [7, 10], [3, 11], [4, 12], [1, 13], [2, 14]]  # mix time and costs and ASCs
        elseif C == 1016
            mix_inds = [[6, 9], [7, 10], [1, 11], [2, 12]]  # mix time and ASCs
        elseif C == 1017
            mix_inds = [[1, 9], [2, 10]]  # mix ASCs only
        elseif C == 1018
            mix_inds = [[1, 9]]  # mix ASC Car only
        end
    end
    if D == 4 
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 1, 0, 0, 0
        betalen = 5
		J = 4
	    if C == 1010
            mix_inds = [[5, 6]]  # mix just time
        elseif C == 1011
            mix_inds = [[5, 6], [4, 7]] # mix time and costs
        elseif C == 1012
            mix_inds = [[4, 6]] # mix costs
        elseif C == 1018
            mix_inds = [[2, 6]]  # mix ASC Car
        end
    end
    if D == 5 
		J = 5
        Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM = 0, 0, 0, 1, 0, 0
        betalen = 5
        if C == 1010
            mix_inds = [[5, 6]]  # mix cost
        elseif C == 1011
            mix_inds = [[5, 6], [1, 7], [2, 8], [3, 9], [4, 10]]  # mix cost and ASCs
        elseif C == 1012
            mix_inds = [[1, 6], [2, 7], [3, 8], [4, 9]]  # mix ASCs
		end
    end

    # ok so mentally we have extended x by len(mix_inds)
    # now we should extend it based on the latent class...

    # there should be an easier way to do it no?
    # yes. with the extra_inds connector

    class_3_ks = nothing
    extra_inds = []
    mix_inds = []

    class_1_av = collect(1:J) 
    class_2_av = collect(1:J) 
    class_3_av = collect(1:J)

    if D == 1
        # 1 ASC rail
        # 2 Cost
        # 3 Time
        if C == 102
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2] 
            prob_inds = [4] 
        elseif C == 1022
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 3] 
            prob_inds = [4] 
        elseif C == 1023
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1] 
            prob_inds = [4] 
        elseif C == 1029
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2, 4] 
            prob_inds = [5] 
            extra_inds = [[3, 4]]
        elseif C == 10229
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 4, 5] 
            prob_inds = [6] 
            extra_inds = [[2, 4], [3, 5]]
        elseif (C == 1028) || (C == 1082)
            class_1_ks = [1, 2, 3] 
            class_2_ks = [2, 3, 4] 
            prob_inds = [5] 
            extra_inds = [[1, 4]]
        elseif C == 103
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2] 
            class_3_ks = [1, 3]
            prob_inds = [4, 5] 
        elseif C == 31
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2] 
            class_3_ks = [1]
            prob_inds = [4, 5] 
        elseif C == 32
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 3] 
            class_3_ks = [1]
            prob_inds = [4, 5] 
        elseif C == 39
            class_1_ks = [1, 2, 3] 
            class_2_ks = [1, 2, 4] 
            class_3_ks = [1, 3, 5]
            prob_inds = [6, 7] 
            extra_inds = [[3, 4], [2, 5]]
        end
    end

    if D == 2
        # ASC_CAR      -0.185907      0.482518    -0.385285      0.700026
        # ASC_TRAIN    -0.330812      0.616437    -0.536651      0.591509
        # B_COST       -1.839381      0.683981    -2.689229      0.007162
        # B_HE         -3.878408      8.934273    -0.434104      0.664213
        # B_TIME  
        if C == 22 
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 5] 
            prob_inds = [6]
        elseif C == 23
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            prob_inds = [6]
        elseif C == 24
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2] 
            prob_inds = [6]
        elseif C == 29
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4, 6] 
            prob_inds = [7]
            extra_inds = [[5, 6]]
        elseif C == 229
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 6, 7] 
            prob_inds = [8]
            extra_inds = [[3, 6], [5, 7]]
		elseif C == 2292
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 5, 6] 
            prob_inds = [7]
            extra_inds = [[3, 6]]
        elseif C == 28
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [3, 4, 5, 6, 7] 
            prob_inds = [8]
            extra_inds = [[1, 6], [2, 7]]
        elseif C == 282
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [2, 3, 4, 5, 6] 
            prob_inds = [7]
            extra_inds = [[1, 6]]
        elseif C == 31 
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4] 
            class_3_ks = [1, 2]
            prob_inds = [6, 7]
        elseif C == 32
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 4, 5] 
            class_3_ks = [1, 2]
            prob_inds = [6, 7]
        elseif C == 33
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            class_3_ks = [1, 2]
            prob_inds = [6, 7]
        elseif C == 39
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4, 6] 
            class_3_ks = [1, 2, 4, 5, 7] 
            prob_inds = [8, 9]
            extra_inds = [[5, 6], [3, 7]]
        elseif C == 1020
			mix_inds = [[5, 6]]  # mix time in class 1
			class_1_ks = [1, 2, 3, 4, 5, 6]
			class_2_ks = [3, 4, 5, 7]
			prob_inds = [8]
			extra_inds = [[2, 7]]
			class_1_av = [1, 2, 3]
			class_2_av = [1, 2]
        elseif C == 1021
	        mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6] # mix
            class_2_ks = [3, 4, 5, 7] # sep ASCs, no SM
            prob_inds = [8]
            extra_inds = [[1, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 3]
        elseif (C == 1022) || (C == 1027)
	        mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6] # mix time
            class_2_ks = [3, 4, 5, 7, 8] # sep ASCs
            prob_inds = [9]
            extra_inds = [[1, 7], [2, 8]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
        elseif (C == 1023) 
	        mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6] # mix time
            class_2_ks = [1, 2, 3, 4, 5] # no car
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2]
        elseif (C == 1024) 
	        mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [1, 2, 3, 4, 5]  # no SM
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3]
            class_2_av = [1, 3]
        elseif C == 1025 
	        mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [1, 2, 3, 4, 7]  # new time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
        elseif C == 1026
	        mix_inds = [[3, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix costs
            class_2_ks = [1, 2, 3, 4, 7]  # new time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
        elseif C == 1028
	        mix_inds = [[4, 6]]  # mix HE in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix HE
            class_2_ks = [1, 2, 3, 4, 7]  # new time
            prob_inds = [8]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
        elseif C == 1030
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [3, 4, 5, 7, 8]  # sep ASCs
            class_3_ks = [1, 2, 3, 4, 5]  # no car
            prob_inds = [9, 10]
            extra_inds = [[1, 7], [2, 8]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
            class_3_av = [1, 2]
        elseif C == 1031
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]  # mix time
            class_2_ks = [3, 4, 5, 7, 8]  # sep ASCs
            class_3_ks = [1, 2, 3, 4, 5]  # no SM
            prob_inds = [9, 10]
            extra_inds = [[1, 7], [2, 8]]
            class_1_av = [1, 2, 3]
            class_2_av = [1, 2, 3]
            class_3_av = [1, 3]
        end
    end

    if D == 3
        # ASC_CAR             0.289219      0.712165     0.406113      0.684660
        # ASC_SM             -0.155131      0.980680    -0.158187      0.874310
        # BETA_COST_HWH     -25.632326     16.309223    -1.571646      0.116033
        # BETA_COST_OTHER    -5.226712      4.526484    -1.154696      0.248215
        # BETA_DIST          -2.925026      2.265154    -1.291314      0.196595
        # BETA_TIME_CAR     -21.029007     24.024233    -0.875325      0.381397
        # BETA_TIME_PT       -8.713693     10.444978    -0.834247      0.404142
        # BETA_WAITING_TIME  -0.074130      0.149418    -0.496126      0.619806
        if C == 22
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 5, 6, 7, 8]
            prob_inds = [9]
        elseif C == 23
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2]
            prob_inds = [9]
        elseif C == 29
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 3, 4, 5, 8, 9, 10]
            prob_inds = [11]
            extra_inds = [[7, 9], [6, 10]]
        elseif C == 229
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 5, 8, 9, 10, 11, 12]
            prob_inds = [13]
            extra_inds = [[3, 9], [4, 10], [7, 11], [6, 12]]
        elseif C == 28
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [3, 4, 5, 6, 7, 8, 9, 10]
            prob_inds = [11]
            extra_inds = [[1, 9], [2, 10]]
        elseif C == 282
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [2, 3, 4, 5, 6, 7, 8, 9]
            prob_inds = [10]
            extra_inds = [[1, 9]]
        elseif C == 31
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 3, 4, 5]
            class_3_ks = [1, 2]
            prob_inds = [9, 10]
	    elseif C == 32
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 5, 6, 7, 8]
            class_3_ks = [1, 2]
            prob_inds = [9, 10]
        elseif C == 39
            class_1_ks = [1, 2, 3, 4, 5, 6, 7, 8]
            class_2_ks = [1, 2, 3, 4, 5, 8, 9, 10]
            class_3_ks = [1, 2, 5, 6, 7, 8, 11, 12]
            prob_inds = [13, 14]
            extra_inds = [[7, 9], [6, 10], [3, 11], [4, 12]]
        end
    end
    
    if D == 4
        # ASC_Bike      -4.195427      0.777725    -5.394489  6.871879e-08
        # ASC_Car       -1.700855      0.609906    -2.788715  5.291762e-03
        # ASC_PB        -0.688697      0.397275    -1.733551  8.299782e-02
        # Beta_cost     -0.165182      0.072914    -2.265444  2.348545e-02
        # Beta_time     -6.807629      1.512680    -4.500375  6.783375e-06
        if C == 2
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4, 6]
            extra_inds = [[5, 6]] 
            prob_inds = [7] 
        elseif C == 21
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4] 
            prob_inds = [6] 
        elseif C == 22
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            prob_inds = [6] 
        elseif C == 23
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3] 
            prob_inds = [6] 
        elseif C == 29
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4, 6] 
            prob_inds = [7]
            extra_inds = [[5, 6]] 
        elseif C == 229
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 6, 7] 
            prob_inds = [8]
            extra_inds = [[4, 6], [5, 7]] 
        elseif C == 28
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [4, 5, 6, 7, 8] 
            prob_inds = [9]
            extra_inds = [[1, 6], [3, 7], [2, 8]] 
        elseif C == 282
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 3, 4, 5, 6] 
            prob_inds = [7]
            extra_inds = [[2, 6]] 
        elseif C == 3
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4] 
            class_3_ks = [1, 2, 3, 5]
            prob_inds = [6, 7] 
        elseif C == 31
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4]
            class_3_ks = [1, 2, 3]
            prob_inds = [6, 7] 
        elseif C == 32
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 5] 
            class_3_ks = [1, 2, 3]
            prob_inds = [6, 7] 
        elseif C == 39
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4, 6] 
            class_3_ks = [1, 2, 3, 5, 7] 
            prob_inds = [8, 9] 
            extra_inds = [[4, 7], [5, 6]] 
        elseif C == 1020
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8]
            prob_inds = [9]
            extra_inds = [[1, 7], [3, 8]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3]
        elseif C == 1021
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7]
            prob_inds = [8]
            extra_inds = [[3, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [3, 4]
        elseif (C == 1022) || (C == 1027)
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8, 9]
            prob_inds = [10]
            extra_inds = [[1, 7], [2, 8], [3, 9]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
        elseif C == 1023
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 5]
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3] # no car
        elseif C == 1024
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 5]
            prob_inds = [7]
            extra_inds = []
            class_1_av = [1, 2, 3, 4]
            class_2_av = [3, 4] # lazy
        elseif C == 1025
            mix_inds = [[4, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 7]  # only mean costs, new beta time
            prob_inds = [8]
            extra_inds = [[5, 7]] # adding beta time
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
        elseif C == 1026
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 5, 7]  # new beta cost
            prob_inds = [8]
            extra_inds = [[4, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
        elseif C == 1030
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8, 9]
            class_3_ks = [1, 2, 3, 4, 5]
            prob_inds = [10, 11]
            extra_inds = [[1, 7], [2, 8], [3, 9]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [1, 2, 3]
        elseif C == 1031
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [4, 5, 7, 8, 9]
            class_3_ks = [1, 2, 3, 4, 5]
            prob_inds = [10, 11]
            extra_inds = [[1, 7], [2, 8], [3, 9]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [3, 4]
        elseif C == 1032
            mix_inds = [[4, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 7]  # only mean costs, new beta time
            class_3_ks = [1, 2, 3, 4, 5]  # base model
            prob_inds = [8, 9]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [3, 4]
        elseif C == 1033
            mix_inds = [[4, 6]]  # mix costs in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 4, 7]  # only mean costs, new beta time
            class_3_ks = [1, 2, 3, 4, 5]  # base model
            prob_inds = [8, 9]
            extra_inds = [[5, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [1, 2, 3]
        elseif C == 1034
            mix_inds = [[5, 6]]  # mix time in class 1
            class_1_ks = [1, 2, 3, 4, 5, 6]
            class_2_ks = [1, 2, 3, 5, 7]  # new beta cost
            class_3_ks = [1, 2, 3, 4, 5]  # lazy
            prob_inds = [8, 9]
            extra_inds = [[4, 7]]
            class_1_av = [1, 2, 3, 4]
            class_2_av = [1, 2, 3, 4]
            class_3_av = [3, 4]
        end
    end

    if D == 5
        # ASC_BM       -7.890775e-01  3.266113e-01 -2.415953e+00  1.569410e-02
        # ASC_EF        8.590257e+00  1.018424e+00  8.434856e+00  0.000000e+00
        # ASC_LF        1.085781e+00  3.054990e-01  3.554123e+00  3.792417e-04
        # ASC_MF        1.430229e+00  5.636678e-01  2.537362e+00  1.116913e-02
        # B_COST       -1.988688e+00  4.028596e-01 -4.936429e+00  7.956610e-07
        if C == 229
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4, 6] 
            prob_inds = [7] 
            extra_inds = [[5, 6]] 
        elseif C == 28
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [5, 6, 7, 8, 9] 
            prob_inds = [10] 
            extra_inds = [[1, 6], [2, 7], [3, 8], [4, 9]]
        elseif C == 282
            class_1_ks = [1, 2, 3, 4, 5] 
            class_2_ks = [1, 2, 3, 4] 
            prob_inds = [6] 
        end
    end

    betalen = prob_inds[end]

    # Run warm-up to ensure the code is compiled
    warm_up_nat_latent(C, D, class_1_ks, class_2_ks, class_3_ks, extra_inds, prob_inds, mix_inds)

    if string(C)[1] == '2' || string(C)[3] == '2'
        actualC = 2
    elseif string(C)[1] == '3' || string(C)[3] == '3'
        actualC = 3
    end

    # Initialize arrays to store results
    # total_LL_1 = 0.0
    # total_LL_2 = 0.0
    # total_bhamsle_sLL = 0.0
    # total_bhamsle_LL = 0.0
    # total_bioTime_1 = 0.0
    # total_bhamsleTime = 0.0
    # total_bioTime_2 = 0.0
    # total_bio_FirstBeta = zeros(betalen)
	# total_bhamsle_beta = zeros(betalen)
	# total_bio_SecondBeta = zeros(betalen)
	# total_bio_FirstProbs = zeros(actualC)
	# total_bhamsle_probs = zeros(actualC)
	# total_bio_SecondProbs = zeros(actualC) 

    bioR = R
    # this is for highR experiments
    if R > 1000
        bioR = 1000
    end

    # this is because I think simulation is fast
    simR = 10000

    for pandaSeed in seed_start:seed_end
        # Run Biogeme and BHAMSLE for the first estimation
		# bio_start_beta_null = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        FirstbioBeta, FirstbioTime, FirstbioLL, FirstbioProbs = run_Biogeme_Latent(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, nothing)
        bhamsle_sLL, bhamsleBeta, bhamsleTime, old_x, old_y, old_av = run_BHAMSLE_Latent(N, R, C, D, FirstbioBeta, pandaSeed, class_1_ks, class_2_ks, class_3_ks, extra_inds, prob_inds, mix_inds, class_1_av, class_2_av, class_3_av)

        # Apply permutation and compute probabilities P1, P2, P3
        bio_start_beta = copy(bhamsleBeta)

        if actualC == 2
            x = bio_start_beta[end]
            P1, P2 = x, 1 - x
            bio_start_beta[end] = round(log(P1) - log(P2), digits=3)
        elseif actualC == 3
            x, y = bio_start_beta[end-1:end]
            P1, P2, P3 = x, y - x, 1 - y
            bio_start_beta[end-1:end] .= [round(log(P1) - log(P3), digits=3), round(log(P2) - log(P3), digits=3)]
        end
		
		# println(class_1_ks)
		# println(class_2_ks)
		# println(class_3_ks)
		
        # Compute log-likelihoods
        if 1000 <= C <= 1999
            First_Bio_LL = simulate_likelihood_mixed(N, D, pandaSeed, FirstbioBeta, C, simR)
            bhamsle_LL = simulate_likelihood_mixed(N, D, pandaSeed, bio_start_beta, C, simR)
        elseif actualC == 2 # should only be accessed in logit latent
            First_Bio_LL = compute_LL_latent2classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, biogeme=true)
            bhamsle_LL = compute_LL_latent2classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, biogeme=false)
        elseif actualC == 3
            First_Bio_LL = compute_LL_latent3classes(old_x, old_y, old_av, FirstbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
            bhamsle_LL = compute_LL_latent3classes(old_x, old_y, old_av, bhamsleBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=false)
		end
		
        println("FirstbioLL = ", FirstbioLL)
		println("First_Bio_LL = ", First_Bio_LL)
        println("FirstbioBeta = ", FirstbioBeta)
        println("FirstbioTime = ", FirstbioTime)
		
		println("bhamsle_sLL = ", bhamsle_sLL)
		println("bhamsle_LL = ", bhamsle_LL)
		println("bhamsleBeta = ", bhamsleBeta)
        println("bhamsleTime = ", bhamsleTime)
		flush(stdout)
		

        # Run Biogeme for the second estimation
        SecondbioBeta, SecondbioTime, SecondbioLL, SecondbioProbs = run_Biogeme_Latent(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, bio_start_beta)
        
        if 1000 <= C <= 1999
            Second_Bio_LL = simulate_likelihood_mixed(N, D, pandaSeed, SecondbioBeta, C, simR)
        elseif actualC == 2 # should only be accessed in logit latent
            Second_Bio_LL = compute_LL_latent2classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, biogeme=true)
        elseif actualC == 3
            Second_Bio_LL = compute_LL_latent3classes(old_x, old_y, old_av, SecondbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
		end

        println("SecondbioLL = ", SecondbioLL)
		println("Second_Bio_LL = ", Second_Bio_LL)
        println("Second Bio Improvement = ", round(((First_Bio_LL - Second_Bio_LL)/First_Bio_LL) * 100,digits=2))
        println("SecondbioBeta = ", SecondbioBeta)
        println("SecondbioTime = ", SecondbioTime)
		flush(stdout)
		
		if C == 1027 || C == 1032
			continue 
		end

        # Define initial guess for beta and initial step size (sigma)
        if !isnothing(extra_inds)
            E = length(extra_inds)
        else
            E = 0
        end
        H = length(mix_inds)
        K = 5 # holds for both SM and LPMC

        initial_beta = Float64.(vcat(fill(0, K), fill(1, H), fill(0, E), fill(0, actualC-1)))
        # initial_beta = ones(length(FirstbioBeta))
        # LB = minimum([- maximum(abs.(FirstbioBeta)) * 1.5, -12])
        # UB = maximum([maximum(abs.(FirstbioBeta)) * 1.5, 12])
		
        LB = -100.0
        UB = 100.0 
		
        ssigma = Float64((UB-LB) / 3) 
		# dim = length(initial_beta)
		# λ_value = 4 + floor(Int, 10 * log(dim))
		# λ_value = max(λ_value, 10)  # Ensure a minimum population size, e.g., 10
        λ_value = Int(50) 

        timeCMA_ES = @elapsed begin
            CMA_ES_beta, CMA_ES_LL_orig = find_optimal_full_beta_cmaes(N, C, actualC, D, pandaSeed, R, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, initial_beta, LB, UB; λ=λ_value, sigma0=ssigma)
        end
		
		if 1000 <= C <= 1999
			CMA_ES_LL = simulate_likelihood_mixed(N, D, pandaSeed, CMA_ES_beta, C, simR)
		elseif actualC == 2 # should only be accessed in logit latent
			CMA_ES_LL = compute_LL_latent2classes(old_x, old_y, old_av, CMA_ES_beta, class_1_ks, class_2_ks, biogeme=true)
		elseif actualC == 3
			CMA_ES_LL = compute_LL_latent3classes(old_x, old_y, old_av, CMA_ES_beta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
		end
		
		println("CMA_ES_LL_orig = ", CMA_ES_LL_orig)
		println("CMA_ES_LL = ", CMA_ES_LL)
		println("CMA_ES_Beta = ", CMA_ES_beta)
        println("CMA_ES_Time = ", timeCMA_ES)
		flush(stdout)

        # Run Biogeme for the third estimation

        # scale beta to avoid overflow in Biogeme exp
        cap_value = -Inf
        if any(x -> x > cap_value, CMA_ES_beta)
			# clamped_CMA_ES_beta = clamp.(CMA_ES_beta, -Inf, cap_value)
            scaled_CMA_ES_beta = scale_to_range(CMA_ES_beta, (LB, UB))
        else
            scaled_CMA_ES_beta = CMA_ES_beta
        end

        # println("Scaled_CMA_ES_beta: ", scaled_CMA_ES_beta)

        startbeta2 = copy(scaled_CMA_ES_beta)
        expCutOff = 20
        if actualC == 2
            if startbeta2[end] > expCutOff
                startbeta2[end] = expCutOff
            end
        elseif actualC == 3
            if startbeta2[end] > expCutOff
                startbeta2[end] = expCutOff
            end
            if startbeta2[end-1] > expCutOff
                startbeta2[end-1] = expCutOff
            end
        end

        println("startbeta2 = ", startbeta2)

        ThirdbioBeta, ThirdbioTime, ThirdbioLL, ThirdbioProbs = run_Biogeme_Latent(N, bioR, C, pandaSeed, Mseed, Tseed, MseedL, TseedL, MseedSM, TseedSM, startbeta2)
        
        if 1000 <= C <= 1999
            Third_Bio_LL = simulate_likelihood_mixed(N, D, pandaSeed, ThirdbioBeta, C, simR)
        elseif actualC == 2 # should only be accessed in logit latent
            Third_Bio_LL = compute_LL_latent2classes(old_x, old_y, old_av, ThirdbioBeta, class_1_ks, class_2_ks, biogeme=true)
        elseif actualC == 3
            Third_Bio_LL = compute_LL_latent3classes(old_x, old_y, old_av, ThirdbioBeta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
		end

        println("ThirdbioLL = ", ThirdbioLL)
		println("Third_Bio_LL = ", Third_Bio_LL)
        println("Third Bio Improvement = ", round(((First_Bio_LL - Third_Bio_LL)/First_Bio_LL) * 100,digits=2))
        println("ThirdbioBeta = ", ThirdbioBeta)
        println("ThirdbioTime = ", ThirdbioTime)
		flush(stdout)

        # total_LL_1 += First_Bio_LL
        # total_LL_2 += Second_Bio_LL
        # total_bhamsle_sLL += bhamsle_sLL
        # total_bhamsle_LL += bhamsle_LL
        # total_bioTime_1 += FirstbioTime
        # total_bhamsleTime += bhamsleTime
        # total_bioTime_2 += SecondbioTime
	    # total_bio_FirstBeta .+= FirstbioBeta
		# total_bhamsle_beta .+= bhamsleBeta
		# total_bio_SecondBeta .+= SecondbioBeta

        # if actualC == 2
        #     total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3)]
        #     total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[1]], digits=3)]
        #     total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3)]
        # elseif actualC == 3
        #     total_bio_FirstProbs .+= [round(FirstbioProbs[1], digits=3), round(FirstbioProbs[2], digits=3), round(FirstbioProbs[3], digits=3)]
        #     total_bhamsle_probs .+= [round(bhamsleBeta[prob_inds[1]], digits=3), round(bhamsleBeta[prob_inds[2]]-bhamsleBeta[prob_inds[1]], digits=3), round(1-bhamsleBeta[prob_inds[2]], digits=3)]
        #     total_bio_SecondProbs .+= [round(SecondbioProbs[1], digits=3), round(SecondbioProbs[2], digits=3), round(SecondbioProbs[3], digits=3)]
        # end
    end

    # Compute averages
    # avg_LL_1 = total_LL_1 / (seed_end - seed_start + 1)
    # avg_LL_2 = total_LL_2 / (seed_end - seed_start + 1)
    # avg_bhamsle_sLL = total_bhamsle_sLL / (seed_end - seed_start + 1)
    # avg_bhamsle_LL = total_bhamsle_LL / (seed_end - seed_start + 1)
    # avg_bioTime_1 = total_bioTime_1 / (seed_end - seed_start + 1)
    # avg_bhamsleTime = total_bhamsleTime / (seed_end - seed_start + 1)
    # avg_bioTime_2 = total_bioTime_2 / (seed_end - seed_start + 1)
	
    # avg_bio_FirstBeta = total_bio_FirstBeta ./ (seed_end - seed_start + 1)
	# avg_bhamsle_beta = total_bhamsle_beta ./ (seed_end - seed_start + 1)
	# avg_bio_SecondBeta = total_bio_SecondBeta ./ (seed_end - seed_start + 1)
	
	# avg_bio_FirstProbs = total_bio_FirstProbs ./ (seed_end - seed_start + 1)
	# avg_bhamsle_probs = total_bhamsle_probs ./ (seed_end - seed_start + 1)
	# avg_bio_SecondProbs = total_bio_SecondProbs ./ (seed_end - seed_start + 1)

	# results = Dict(
	#     "N" => N,
	#     "R" => R,
	#     "avg_LL_1" => avg_LL_1,
	#     "avg_bioTime_1" => avg_bioTime_1,
	#     "avg_LL_2" => avg_LL_2,
    #     "avg_bhamsle_sLL" => avg_bhamsle_sLL,
    #     "avg_bhamsle_LL" => avg_bhamsle_LL,
	#     "avg_bhamsleTime" => avg_bhamsleTime,
	#     "avg_bioTime_2" => avg_bioTime_2,
	#     "avg_bio_FirstBeta" => avg_bio_FirstBeta,
	#     "avg_bhamsle_beta" => avg_bhamsle_beta,
	#     "avg_bio_SecondBeta" => avg_bio_SecondBeta,
	#     "avg_bio_FirstProbs" => avg_bio_FirstProbs,
	#     "avg_bhamsle_probs" => avg_bhamsle_probs,
	#     "avg_bio_SecondProbs" => avg_bio_SecondProbs
	# )

    # if D == 1
    #     JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_nether.json", results)
    # elseif D == 2
    #     JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_SM.json", results)
    # elseif D == 3
    #     JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_optima.json", results)
    # elseif D == 4
    #     JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_lpmc.json", results)
    # elseif D == 5
    #     JSON3.write("results_N$(N)_R$(R)_seed$(seed_start)_$(seed_end)_$(C)_telephone.json", results)
    # end
end

# Define your wrapper function
function objective_function(beta, N, C, actualC, D, pandaSeed, R, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks)
    if 1000 <= C <= 1999
        LL = simulate_likelihood_mixed(N, D, pandaSeed, beta, C, R)
    elseif actualC == 2 # should only be accessed in logit latent
        LL = compute_LL_latent2classes(old_x, old_y, old_av, beta, class_1_ks, class_2_ks, biogeme=true)
    elseif actualC == 3
        LL = compute_LL_latent3classes(old_x, old_y, old_av, beta, class_1_ks, class_2_ks, class_3_ks, biogeme=true)
    end

    return -LL  # CMA-ES minimizes, so return negative log-likelihood
end

# Set up the CMA-ES optimization for the full beta vector
function find_optimal_full_beta_cmaes(N, C, actualC, D, pandaSeed, R, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, startbeta, LB, UB; 
                                      λ::Int = 20, sigma0::Float64 =1.0, max_iters::Int = 500)
    # Initial guess for beta vector (starting in the middle of the bounds)
    initial_beta = startbeta  # Element-wise midpoint between LB and UB

    # Set optimization options (e.g., number of iterations)
    options = Evolutionary.Options(iterations=max_iters)  # Specify maximum iterations

    # Ensure bounds are vectors
    lb = fill(LB, length(startbeta))
    ub = fill(UB, length(startbeta))

    # Perform the optimization using CMA-ES for the full beta vector
    result = Evolutionary.optimize(
        beta -> bounded_objective_function(beta, N, C, actualC, D, pandaSeed, R, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks, lb, ub),
        # beta -> objective_function(beta, N, C, actualC, D, pandaSeed, simR, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks),  # Full beta objective
        initial_beta,  # Initial guess for beta vector
        CMAES(λ=λ, sigma0=sigma0),  # CMA-ES with population size and step size
        options  # Optimization options
    )
    
    # Extract the best beta vector and the best objective (negated to undo minimization)
    best_beta = Evolutionary.minimizer(result)
    best_objective = -Evolutionary.minimum(result)  # Negate back to get maximized objective
    
    return best_beta, best_objective  # Return the optimized beta vector and objective
end;

function bounded_objective_function(
    beta::Vector{Float64}, N, C, actualC, D, pandaSeed, R, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks,
    lb::Vector{Float64}, ub::Vector{Float64}, penalty_factor::Float64 = 1e6
)
    # Compute the original objective value
    obj_value = objective_function(beta, N, C, actualC, D, pandaSeed, R, old_x, old_y, old_av, class_1_ks, class_2_ks, class_3_ks)
    
    # Compute penalties for out-of-bounds values
    penalty = sum((beta .< lb) .* penalty_factor .* abs.(lb .- beta)) +
              sum((beta .> ub) .* penalty_factor .* abs.(beta .- ub))
    
    return obj_value + penalty
end


function scale_to_range(vec, target_range=(-1.0, 1.0))
    min_target, max_target = target_range
    max_abs_val = maximum(abs.(vec))  # Get the largest absolute value
    if max_abs_val == 0
        return fill(0.0, length(vec))  # Avoid division by zero
    end
    scaled_vec = (vec ./ max_abs_val) * (max_target - min_target)
    return scaled_vec
end


redirect_stderr(devnull) do
# Parse command-line arguments
N = parse(Int, ARGS[1])
R = parse(Int, ARGS[2])
C = parse(Int, ARGS[3])
D = parse(Int, ARGS[4]) 
# 1 = nether
# 2 = SM
# 3 = Optima
# 4 = LPMC
# 5 = telephone

seed_start = parse(Int, ARGS[5])
seed_end = parse(Int, ARGS[6])

highR = false
try
    global highR = Bool(parse(Int, ARGS[7]))
catch e
end


try
	if (10 <= C <= 19) # idk lets use that to indicate mixed
	    main_mixed(N, R, C, D, seed_start, seed_end, highR)
	else # just use 1000 to 1019 and 1020 to 1029 to access mixed latent
	    main_nat_latent(N, R, C, D, seed_start, seed_end)
	end
catch e
    ""
end




# if C == 3
#     # main_3_LPMC(N, R, seed_start, seed_end)
#     # main_3_SM(N, R, seed_start, seed_end)
#     main_3_optima(N, R, seed_start, seed_end)
# elseif C == 2
#     # main_2(N, R, seed_start, seed_end)
#     # main_2_SM(N, R, seed_start, seed_end)
#     main_2_optima(N, R, seed_start, seed_end)
# elseif C == 22
#     # main_2(N, R, seed_start, seed_end)
#     # main_2_SM(N, R, seed_start, seed_end)
#     main_22_optima(N, R, seed_start, seed_end)
# elseif C == 4
#     # main_4(N, R, seed_start, seed_end)
#     main_4_SM(N, R, seed_start, seed_end)
# elseif C == 5
#     main_5_SM(N, R, seed_start, seed_end)
# elseif C == 10 # idk lets use that to indicate mixed, and SM be first
#     main_mixed_SM(N, R, seed_start, seed_end)
# end
end
