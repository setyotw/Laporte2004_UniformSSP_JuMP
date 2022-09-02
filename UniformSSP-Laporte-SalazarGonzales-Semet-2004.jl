## Problem: Job sequencing and tool switching problem (SSP) - Laporte, Salazar-Gonzales, Semet (2004) formulation
## Solver: Gurobi
## Language: Julia (JuMP)
## Written by: @setyotw
## Date: Sept 2, 2022

#%% import packages
using Pkg, JuMP, Gurobi, DataStructures, Combinatorics
Pkg.status()
import FromFile: @from

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  DEVELOPMENT PARTS
function UniformSSP_Laporte_Formulation(instanceSSP, magazineCap, MILP_Limit)
    # 1 | initialize sets and notations
    # number of available jobs (horizontal)
    n = length(instanceSSP[1,:])
    # number of available tools (vertical)
    m = length(instanceSSP[:,1])

    J = [i for i in range(1, n)] ## list of jobs
    J0 = [i for i in range(0, n)] ## list of positions
    T = [i for i in range(1, m)] ## list of tools
    arcIJ = [(i,j) for i in J0 for j in J0]
    arcJT = [(j,t) for j in J for t in T]
    
    Tj = Dict((j) => [] for j in J)
    TnotTj = Dict((j) => [] for j in J)
    for job in J
        for tools in T
            if instanceSSP[tools,job] == 1
                append!(Tj[job], tools)
            else
                append!(TnotTj[job], tools)
            end
        end
    end

    # 2 | initialize parameters
    C = Int(magazineCap)
    
    # 3 | initialize the model
    model = Model(Gurobi.Optimizer)

    # 4 | initialize decision variables
    @variable(model, X[arcIJ], Bin) # U[jk] = Equal to 1 if job j processed in position k
    @variable(model, Y[arcJT], Bin) # V[kt] = Equal to 1 if tool t presents while performing a job in position k
    @variable(model, Z[arcJT], Bin) # W[kt] = tool switch, equal to 1 if tool t is in magazine while performing a job
    
    # 5 | define objective function
    @objective(model, Min, 
        sum(Z[(j,t)] for j in J for t in Tj[j]))

    # 6 | define constraints
    # (1) sum-j-in-range-J U 0 (j/=i) Xij = 1, for i in range J U 0  ##
    for i in J0
        @constraint(model, sum(X[(i,j)] for j in J0 if j!=i) == 1)
    end
    
    # (2) sum-i-in-range-J U 0 (i/=j) Xij = 1, for j in range J U 0  ##
    for j in J0
        @constraint(model, sum(X[(i,j)] for i in J0 if i!=j) == 1)
    end

    # (3) sum-i,j-in-range-S Xij <= |S| - 1, for S in J U 0 and 2 <= |S| <= n-1 ##
    subtourSet = collect(powerset(J0))[2:end-1]
    for S in subtourSet
        @constraint(model, sum(X[(i,j)] for i in S for j in S if i!=j) <= length(S)-1)
    end

    # (4) sum-t-in-range-T Yjt <= C, for j in range J
    for j in J
        @constraint(model, sum(Y[(j,t)] for t in T) <= C)
    end

    # (5) Xij + Yjt - Yit <= Zjt + 1, for i in range J, for j in range J, i!=j, for t in range T
    for i in J
        for j in J
            for t in T
                if i!=J
                    @constraint(model, X[(i,j)]+Y[(j,t)]-Y[(i,t)] <= Z[(j,t)]+1)
                end
            end
        end
    end

    # (6) X0j + Yjt <= Zjt + 1, for j in range J, for t in range T
    for j in J
        for t in T
            @constraint(model, X[(0,j)]+Y[(j,t)] <= Z[(j,t)]+1)
        end
    end

    # (7) Yjt = 1, for j in range J and t in range Tj
    for j in J 
        for t in Tj[j]
            @constraint(model, Y[(j,t)] == 1)
        end
    end

    # (8) Zit = 0, for i in range J and t in range T(T/=Tj) 
    for j in J 
        for t in TnotTj[j]
            @constraint(model, Z[(j,t)] == 0)
        end
    end

    # 7 | call the solver (we use Gurobi here, but you can use other solvers i.e. PuLP or CPLEX)
    JuMP.set_time_limit_sec(model, MILP_Limit)
    JuMP.optimize!(model)

    # 8 | extract the results    
    completeResults = solution_summary(model)
    solutionObjective = objective_value(model)
    solutionGap = relative_gap(model)
    runtimeCount = solve_time(model)
    all_var_list = all_variables(model)
    all_var_value = value.(all_variables(model))
    X_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'X']
    Y_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'Y']
    Z_active = [string(all_var_list[i]) for i in range(1,length(all_var_list)) if all_var_value[i] > 0 && string(all_var_list[i])[1] == 'Z']
    
    return solutionObjective, solutionGap, X_active, Y_active, Z_active, runtimeCount, completeResults
end

#%%%%%%%%%%%%%%%%%%%%%%%%%%
#  IMPLEMENTATION PARTS
#%% input problem instance
# a simple uniform SSP case with 5 different jobs, 6 different tools, and 3 capacity of magazine (at max, only 3 different tools could be installed at the same time)
instanceSSP = Array{Int}([
        1 1 0 0 1;
        1 0 0 1 0;
        0 1 1 1 0;
        1 0 1 0 1;
        0 0 1 1 0;
        0 0 0 0 1])

magazineCap = Int(3)

#%% termination time for the solver (Gurobi)
MILP_Limit = Int(3600)

#%% implement the mathematical formulation
# solutionObjective --> best objective value found by the solver
# solutionGap --> solution gap, (UB-LB)/UB
# U_active, V_active, W_active --> return the active variables
# runtimeCount --> return the runtime in seconds
# completeResults --> return the complete results storage
solutionObjective, solutionGap, U_active, V_active, W_active, runtimeCount, completeResults = UniformSSP_Laporte_Formulation(instanceSSP, magazineCap, MILP_Limit)