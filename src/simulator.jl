module Simulator

using LinearAlgebra, StaticArrays, Interpolations
using ProgressMeter, RecipesBase
# using ..EnvData, ..NGPKF, ..ErgodicController, ..Ergodic2
using ..EnvData
using CoverageControllers, SpatiotemporalGPs

using JLD2: JLD2, jldopen
using FileIO: load

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
    
    wx_hat::M  # estimated wind speed x
    wy_hat::M  # .. y
    wx_q::M    # clarity in x
    wy_q::M    # clarity in y
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


    # save the new maps
    w_ts = [t0]
    wx_hats = [STGPKF.get_estimate(stgpkf_problem, wx_state)]
    wy_hats = [STGPKF.get_estimate(stgpkf_problem, wy_state)]
    wx_qs = [STGPKF.get_estimate_clarity(stgpkf_problem, wx_state)]
    wy_qs = [STGPKF.get_estimate_clarity(stgpkf_problem, wy_state)]

    # create a clarity map
    # get the average clarity map (should not do anything in principle, since both of the clarity maps should be same)
    q_map = 0.5 * (wx_qs[end] + wy_qs[end])
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


    @showprogress for (it, t) in enumerate(ts[2:end])

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

          # save the new maps
          push!(w_ts, t)
          push!(wx_hats, STGPKF.get_estimate(stgpkf_problem, wx_state))
          push!(wy_hats, STGPKF.get_estimate(stgpkf_problem, wy_state))
          push!(wx_qs, STGPKF.get_estimate_clarity(stgpkf_problem, wx_state))
          push!(wy_qs, STGPKF.get_estimate_clarity(stgpkf_problem, wy_state))

          last_assimilate_time = t
          empty!(tmp_measurements) # reset all the measurements
        end

        # get the clarity map
        q_map = 0.5 * (wx_qs[end] + wy_qs[end])
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

    return SimResult(ts, xs, us, w_ts, wx_hats, wy_hats, wx_qs, wy_qs)

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



end
