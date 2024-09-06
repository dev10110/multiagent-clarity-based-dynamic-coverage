module Simulator

using LinearAlgebra, StaticArrays, Interpolations
using ProgressMeter, RecipesBase
# using ..EnvData, ..NGPKF, ..ErgodicController, ..Ergodic2
using ..EnvData
using CoverageControllers, SpatiotemporalGPs

using JLD2: JLD2, jldopen
using FileIO: load
using CodecZlib # for compression

struct Measurement{T,P,F}
    t::T
    p::P
    y::F
end

"""
  take_measurement(t, p, data; σ_meas = 0, Q_meas = σ_meas*I)

returns a SVector of the [wx, wy] at time t, and pos p by querying the data 
"""
function measure(
    t,
    p::SV,
    data::WD;
    σ_meas = 0,
    Q_meas = σ_meas * I,
) where {WD<:WindData,SV<:Union{SVector{2},NTuple{2,Float64}}}

    y = data(p..., t) + Q_meas * randn(2)
    sp = SVector{2}(p...)
    return Measurement(t, sp, y)

end

function measure(
    t,
    ps::VSV,
    data::WD;
    σ_meas = 0,
    Q_meas = σ_meas * I,
) where {WD<:WindData,SV<:Union{SVector{2},NTuple{2,Float64}},VSV<:AbstractVector{SV}}

    return [measure(t, p, data; Q_meas = Q_meas) for p in ps]

end

struct SimResult{T,X,U,VT, VM}
    ts::T
    xs::X
    us::U
    #
    w_ts::VT # times at which the maps were updated
    wx_hats::VM
    wy_hats::VM
    wx_qs::VM
    wy_qs::VM
end


struct SimState{T, X, U, M}
    t::T       # current time
    xs::X      # robot positions
    us::U      # robot control inputs
    
    # wx_state::M  # estimated wind speed x
    # wy_state::M  # .. y
    wx_hat::M
    wy_hat::M
    w_q::M
end



function domain_bounds(coverage_grid::CoverageControllers.Grid{D}) where {D}

  lower = coverage_grid.o
  L = CoverageControllers.lengths(coverage_grid)
  upper = lower .+ L

  return (lower, upper)

end


function Base.clamp(p, coverage_grid::CoverageControllers.Grid{D}; margin=0.0) where {D}

  lower, upper = domain_bounds(coverage_grid)

  new_p = ntuple( i -> clamp(p[i], lower[i]+margin, upper[i]-margin), D)

  return new_p
end

function simulate(
    ts,
    x0,
    controllers,
    windData::EnvData.WindData,
    coverage_grid::CoverageControllers.Grid,
    stgpkf_problem::STGPKFProblem,
    robot_model::CoverageControllers.AbstractRobot,
    filepath # where to save data
    ;
    σ_meas = 0.0
)

    # extract info from arguments
    t0 = ts[1]
    xs = [x0]
    N_robots = length(x0)
    ΔT = Base.step(ts)

    # create the trajectory history 
    traj_history = vcat(xs...)

    # create initial stgpkf states
    wx_state = stgpkf_initialize(stgpkf_problem)
    wy_state = stgpkf_initialize(stgpkf_problem) 

    # get a measurement
    ys = measure(t0, x0, windData; σ_meas = σ_meas)
    @assert length(ys) == N_robots
    
    MeasurementType = eltype(ys)
    tmp_measurements = MeasurementType[]
    last_assimilate_time = t0

    # run the fusion
    measurement_pos = [m.p for m in ys]
    measurements_wx = [m.y[1] for m in ys]
    measurements_wy = [m.y[2] for m in ys]
    Σ_meas = σ_meas^2 * I(length(ys))
    wx_state = STGPKF.stgpkf_correct(
        stgpkf_problem,
        wx_state,
        measurement_pos,
        measurements_wx,
        Σ_meas,
    )
    wy_state = STGPKF.stgpkf_correct(
        stgpkf_problem,
        wy_state,
        measurement_pos,
        measurements_wx,
        Σ_meas,
    )

    # create a clarity map
    # get the average clarity map (should not do anything in principle, since both of the clarity maps should be same)
    q_map = STGPKF.get_estimate_clarity(stgpkf_problem, wx_state)

    # sanity check dimensions are compatible
    @assert length(q_map) == prod(coverage_grid.N)

    # construct the ergodic q map
    coverage_q_map = reshape(q_map, coverage_grid.N)

    # decide the control input for the first step
    u0 = controllers(
        t0,
        x0;
        coverage_grid = coverage_grid,
        coverage_q_map = coverage_q_map,
        traj_history = traj_history,
        robot_model = robot_model,
    )
    us = [u0]

    # todo: write setup to JLD file
    save_setup(filepath, stgpkf_problem, coverage_grid, ts, N_robots)
          
    wx_hat = STGPKF.get_estimate(stgpkf_problem, wx_state)
          wy_hat = STGPKF.get_estimate(stgpkf_problem, wy_state)
          q_map = STGPKF.get_estimate_clarity(stgpkf_problem, wx_state)

          sim_state = SimState(t0, x0, u0, wx_hat, wy_hat, q_map)
          append_result(filepath, 1, sim_state)


          it = 1
    @showprogress for t in ts[2:end]
        it +=1 

        # first try and move time forwards
        # grab all the robot states and control inputs
        x = xs[end]
        u = us[end]

        # apply the control input to move the robots
        x = [CoverageControllers.dynamics(robot_model, xi, ui, ΔT) for (xi, ui) in zip(x, u)]
        
        # make sure each robot is in the domain
        x = [clamp(xi, coverage_grid) for xi in x]
        
        # push the new state
        push!(xs, x)

        # also push to the trajectory history
        push!(traj_history, x...)

        # make a measurement from each robot
        ys = measure(t, x, windData; σ_meas = σ_meas)

        # add all the to the tmp_measurements vector
        push!(tmp_measurements, ys...)

        if (t > last_assimilate_time + stgpkf_problem.ΔT)

          # println("At t=$t, assimilating $(length(tmp_measurements)) measurements")

          # first predict it forward
          wx_state = STGPKF.stgpkf_predict(stgpkf_problem, wx_state)
          wy_state = STGPKF.stgpkf_predict(stgpkf_problem, wy_state)

          # run the fusion
          measurement_pos = [m.p for m in tmp_measurements]
          measurements_wx = [m.y[1] for m in tmp_measurements]
          measurements_wy = [m.y[2] for m in tmp_measurements]

          Σ_meas = σ_meas^2 * I(length(tmp_measurements))

          wx_state = STGPKF.stgpkf_correct(
              stgpkf_problem,
              wx_state,
              measurement_pos,
              measurements_wx,
              Σ_meas,
          )
          wy_state = STGPKF.stgpkf_correct(
              stgpkf_problem,
              wy_state,
              measurement_pos,
              measurements_wx,
              Σ_meas,
          )

          last_assimilate_time = t
          empty!(tmp_measurements) # reset all the measurements

        end

        # every iteration, save the states
          wx_hat = STGPKF.get_estimate(stgpkf_problem, wx_state)
          wy_hat = STGPKF.get_estimate(stgpkf_problem, wy_state)
          q_map = STGPKF.get_estimate_clarity(stgpkf_problem, wx_state)

          sim_state = SimState(t, x, u, wx_hat, wy_hat, q_map)
          append_result(filepath, it, sim_state)

          # get the clarity map
          coverage_q_map = reshape(q_map, coverage_grid.N)

        # decide the control input for the current step
        u = controllers(
            t,
            x;
            coverage_grid = coverage_grid,
            coverage_q_map = coverage_q_map,
            traj_history = traj_history,
            robot_model = robot_model
        )
        push!(us, u)

    end

    return 

end


function save_setup(fname::AbstractString, stgpkf_problem, coverage_grid, ts, Nr)
    jldopen(fname, "a"; compress=true) do fid
        group = JLD2.Group(fid, "setup")
        group["stgpkf_problem"] = stgpkf_problem
        group["coverage_grid"] = coverage_grid
        group["ts"] = ts
        group["Nr"] = Nr
    end
end

"""
    append_result(fname::AbstractString, gname::String, result::T)

Append a `T` instance to a result file

## Arguments

- `fname`: The name of the result file to be appended to.
- `gname`: The unique `JLD2` group name to be used in the file for grouping the data
  associated with this particular `Result`.
- `result`:  The `T` data to be written to the file.
"""
function append_result(fname::AbstractString, tind::Integer, result::SimState)
    jldopen(fname, "a", compress=true) do fid
        group = JLD2.Group(fid, string(tind))
        group["sim_state"] = result
    end
end


function read_result_file(fname::AbstractString, tind::Integer)
    dat = load(fname, "$(tind)/sim_state")
    return dat
end

function read_result_file(fname::AbstractString, ds::AbstractString)
    dat = load(fname, ds)
    return dat
end

@recipe function f(grid::CoverageControllers.Grid, q_vec::VF) where {F, VF<:AbstractVector{F}}

  xs, ys = axes(grid)
  M = reshape(q_vec, length(xs), length(ys))
  @series begin
    seriestype --> :heatmap
    xlabel --> "x [km]"
    ylabel --> "y [km]"
    
    xs, ys, M'
  end

end


## utilities for plotting

using Plots
function visualize_results(time, filepath, windData)

    sim_ts = Simulator.read_result_file(filepath, "setup/ts")
    stgpkf_problem = Simulator.read_result_file(filepath, "setup/stgpkf_problem")
    coverage_grid = Simulator.read_result_file(filepath, "setup/coverage_grid")

    # determine the time indices
    res_ind = searchsortedlast(sim_ts, time)
    sim_state = read_result_file(filepath, "$(res_ind)/sim_state")
    
    # plot the true windData
    plot1 = plot(windData, time; plottype=:wx, titlefontsize=9)

    robot_positions =  sim_state.xs
    scatter!(first.(robot_positions), last.(robot_positions), label=false, color=:white)
    # compute the default boundary 
    gridBoundary = CoverageControllers.GridBoundary(coverage_grid)
    for b in gridBoundary
        plot!(b; dist=0.75)
    end

    # plot the stgpkf state
    plot2 = plot(coverage_grid, sim_state.wx_hat; clims=(-8,8), cmap=:balance, yticks=:none, ylabel="")

    # plot the clarity
    plot3 = plot(coverage_grid, sim_state.w_q; clims=(0, 1), yticks=:none, ylabel="")

    # plot all
    layout = @layout [a b c]
    plot(plot1, plot2, plot3, layout=layout, size=(1200, 500))
end

function get_trajectories(filepath; tmax=Inf)

    Nr = read_result_file(filepath, "setup/Nr")
    ts = read_result_file(filepath, "setup/ts")

    xs = [Float64[] for i=1:Nr]
    ys = [Float64[] for i=1:Nr]

    # get the max number of elements to plot
    Nt = searchsortedlast(ts, tmax)
    
    @showprogress for i=1:Nt
        simstate = read_result_file(filepath, "$(i)/sim_state")

        x = simstate.xs
        for r = 1:Nr
            push!(xs[r], x[r][1])
            push!(ys[r], x[r][2])
        end
    end

    return ts, xs, ys
end


function plot_trajectories!(filepath; tmax=Inf, fade=false, kwargs...)

    _, xs, ys = get_trajectories(filepath; tmax=tmax)

    Nr = length(xs)

    for r=1:Nr
        if fade
            n = length(xs[r])
            plot!(xs[r], ys[r]; linealpha=range(0.0, 1.0, length=n), kwargs...)
      else
          plot!(xs[r], ys[r]; kwargs...)
      end
    end
    plot!()
end



end
