module Simulator

using LinearAlgebra, StaticArrays, Interpolations
using ProgressLogging
using ..EnvData, ..NGPKF, ..ErgodicController, ..Ergodic2


# MATERN SPATIAL LENGTH SCALE = 1.0 km
# MATERN TEMPORAL LENGTH SCALE = 0.2 * 60 * 24 = 4.8 hrs

struct Measurement{T, P, F}
  t::T
  p::P
  y::F
end

"""
  take_measurement(t, p, data; σ_meas = 0, Q_meas = σ_meas*I)

returns a SVector of the [wx, wy] at time t, and pos p by querying the data 
"""
function measure(t, p::SV, data::WD; σ_meas=0, Q_meas=σ_meas * I) where {WD<:WindData,SV<:SVector{2}}

  y = data(p..., t) + Q_meas * randn(2)
  return Measurement(t, p, y)

end

function measure(t, ps::VSV, data::WD; σ_meas=0, Q_meas = σ_meas * I) where {WD<:WindData,SV<:SVector{2}, VSV <: AbstractVector{SV}}

  return [measure(t, p, data; Q_meas=Q_meas) for p in ps]

end

function step(t, x::X, u::U, ΔT) where {X<:SVector,U<:SVector}

  A = I(2)
  B = ΔT * I(2)

  return A * x + B * u
end


function step(t, xs::XS, us::US, ΔT) where {X<:SVector,U<:SVector,XS<:AbstractVector{X},US<:AbstractVector{U}}

  length(xs) == length(us) || throw(DimensionMismatch())

  N = length(xs)

  return [step(t, xs[i], us[i], ΔT) for i = 1:N]

end


struct SimResult{T,X,U,M,TV,W,EM}
  ts::T
  xs::X
  us::U
  measurements::M
  w_hat_ts::TV
  wx_hats::W
  wy_hats::W
  ergo_q_maps::EM
end

function ErgoGrid(ngpkf_grid::G) where {G<:NGPKF.NGPKFGrid}
  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  dxs = (Base.step(ngpkf_grid.xs), Base.step(ngpkf_grid.ys))
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))

  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)

  return ergo_grid
end

# L = dx * (N-1)

function ErgoGrid(ngpkf_grid::G, Ns) where {G<:NGPKF.NGPKFGrid}

  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))
  dxs = Ls ./ (Ns .- 1)

  ergo_grid = ErgodicController.Grid(origin, dxs, Ls)

  return ergo_grid

end
function Ergo2Grid(ngpkf_grid::G, Ns) where {G<:NGPKF.NGPKFGrid}

  origin = (ngpkf_grid.xs[1], ngpkf_grid.ys[1])
  Ls = (maximum(ngpkf_grid.xs) - minimum(ngpkf_grid.xs), maximum(ngpkf_grid.ys) - minimum(ngpkf_grid.ys))
  dxs = Ls ./ (Ns .- 1)

  ergo_grid = Ergodic2.Grid(origin, dxs, Ls)

  return ergo_grid

end

function ngpkf_to_ergo(ngpkf_grid::G1, ergo_grid::G2, clarity_map) where {G1<:NGPKF.NGPKFGrid,G2<:ErgodicController.Grid}

  Ns = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  if Ns == ergo_grid.N
    return 1.0 * clarity_map
  end

  # interpolate the data
  itp = linear_interpolation((ngpkf_grid.xs, ngpkf_grid.ys), clarity_map, extrapolation_bc=Line())
  ergo_map = itp(ErgodicController.xs(ergo_grid), ErgodicController.ys(ergo_grid))

  return ergo_map

end

function ngpkf_to_ergo2(ngpkf_grid::G1, ergo_grid::G2, clarity_map) where {G1<:NGPKF.NGPKFGrid,G2<:Ergodic2.Grid}

  # interpolate the data
  itp = linear_interpolation((ngpkf_grid.xs, ngpkf_grid.ys), clarity_map, extrapolation_bc=Line())
  ergo_map = itp(Ergodic2.xs(ergo_grid), Ergodic2.ys(ergo_grid))

  return ergo_map

end



function simulate(ts, x0::XS, controllers;
  ngpkf_grid::G,
  windData,
  σ_meas=0, 
  σ_process=0,
  Q_process = σ_process^2 * I,
  fuse_measurements_every_ΔT=5.0,
  recompute_controller_every_ΔT=fuse_measurements_every_ΔT) where {X<:SVector,XS<:AbstractVector{X},G<:NGPKF.NGPKFGrid}

  # extract info from arguments
  t0 = ts[1]
  xs = [x0,]
  N_robots = length(x0)
  ΔT = Base.step(ts)
  Ns_grid = length(ngpkf_grid.xs), length(ngpkf_grid.ys)
  ergo_grid = ErgoGrid(ngpkf_grid, (256, 256))

  # setup map states
  w_hat_ts = [t0,]

  wx_hat = NGPKF.initialize(ngpkf_grid)
  wx_hats = [wx_hat,]

  wy_hat = NGPKF.initialize(ngpkf_grid)
  wy_hats = [wy_hat,]

  # clarity map
  q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat, wy_hat)
  ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
  ergo_q_maps = [ergo_q_map,]

  # get a measurement
  # ys = [measure(t0, x0[i], windData; σ_meas=σ_meas) for i = 1:N_robots]
  measurements = [measure(t0, x0, windData; σ_meas = σ_meas)...]  

  # decide the control input for the first step
  u0 = controllers(t0, x0;
    ngpkf_grid=ngpkf_grid,
    ergo_grid=ergo_grid,
    ergo_q_map=ergo_q_maps[end],
    traj=vcat(xs...),
    ΔT=ΔT,
  )

  us = [u0,]

  last_measurement_fuse_time = t0
  last_measurement_fuse_index = 0

  last_control_update_time = t0

  # try
    @progress for (it, t) in enumerate(ts[1:(end-1)])

      x = xs[end]

      # make a measurement from each robot
      ys = measure(t, x, windData; σ_meas=σ_meas)
      append!(measurements, ys)

      # check if we need to fuse measurements
      if (t - last_measurement_fuse_time) >= fuse_measurements_every_ΔT

        # # collect all the locations we have made measurements
        # measurement_pos = vcat(xs[last_measurement_fuse_index:end]...)
        # measurement_w = vcat(measurements[last_measurement_fuse_index:end]...)

        # # extract x and y components of the measurements
        # measurements_wx = [w[1] for w in measurement_w]
        # measurements_wy = [w[2] for w in measurement_w]

        # run NGPKF
        wx_hat = NGPKF.predict(ngpkf_grid, wx_hats[end]; Q_process=Q_process)
        wy_hat = NGPKF.predict(ngpkf_grid, wy_hats[end]; Q_process=Q_process)
        

        # grab the data again
        new_measurements = measurements[(last_measurement_fuse_index+1):end]
        measurement_pos = [m.p for m in new_measurements]
        measurements_wx = [m.y[1] for m in new_measurements]
        measurements_wy = [m.y[2] for m in new_measurements]

        # run the fusion
        wx_hat_new = NGPKF.correct(ngpkf_grid, wx_hat, measurement_pos, measurements_wx; σ_meas=σ_meas)
        wy_hat_new = NGPKF.correct(ngpkf_grid, wy_hat, measurement_pos, measurements_wy; σ_meas=σ_meas)

        # save the new maps
        push!(w_hat_ts, t)
        push!(wx_hats, wx_hat_new)
        push!(wy_hats, wy_hat_new)

        # update the clarity map
        q_map = NGPKF.clarity_map(ngpkf_grid, wx_hat_new, wy_hat_new)
        ergo_q_map = ngpkf_to_ergo(ngpkf_grid, ergo_grid, q_map)
        push!(ergo_q_maps, ergo_q_map)

        last_measurement_fuse_time = t
        last_measurement_fuse_index = length(measurements)
      end


      # if (t - last_control_update_time >= recompute_controller_every_ΔT)
      #   # chose a control action

        traj = vcat(xs...)

        u = controllers(t, x;
          ngpkf_grid=ngpkf_grid,
          ergo_grid=ergo_grid,
          ergo_q_map=ergo_q_maps[end], # current clarity
          traj=traj,  # list of all points visited by all agents
          ΔT=ΔT,
        )
        
        push!(us, u)

      #   last_control_update_time = t

      # end



      # update 
      u = us[end] # use the last control input
      new_xs = step(t, xs[end], u, ΔT)
      push!(xs, new_xs)

    end

  # catch e
    # println(e)
  # end

  return SimResult(ts, xs, us, measurements, w_hat_ts, wx_hats, wy_hats, ergo_q_maps)


end











end