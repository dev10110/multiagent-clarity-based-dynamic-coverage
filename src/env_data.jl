

module EnvData

using NetCDF, DataFrames, Printf, Dates, Interpolations
using StaticArrays

using RecipesBase

const data_root = "./wind_data"

const T_OFFSET = 1063468800 + 2880 # this is minutes between 0001-01-01 and 2023-01-01

export WindData

struct WindData{V, M, F}
  X::V
  Y::V
  T::V
  WX::M
  WY::M
  itp_wx::F
  itp_wy::F 
end



function WindData(X, Y, T, WX, WY)
  try 
    itp_wx = linear_interpolation((X, Y, T), WX)
    itp_wy = linear_interpolation((X, Y, T), WY)
    return WindData(X, Y, T, WX, WY, itp_wx, itp_wy)
  catch
    return WindData(X, Y, T, WX, WY, nothing, nothing)    
  end
end

function (wd::WindData)(x, y, t)
  return @SVector [wd.itp_wx(x, y, t), wd.itp_wy(x, y, t)]
end

function print_info(fn)
  return ncinfo(fn)
end

function load_axes(fn; starts=[1,1,1], counts=[-1,-1,-1])
  
  # Z = Float64.(ncread(fn, "Z"))
  X = Float64.(ncread(fn, "X", start=[starts[1]], count=[counts[1]] ) / 1000)
  Y = Float64.((ncread(fn, "Y" , start=[starts[2]], count=[counts[2]]) |> reverse) / 1000)
  T = round.(Float64.(ncread(fn, "time"; start=[starts[3]], count=[counts[3]]) .- T_OFFSET))

  return X, Y, T

end

function get_filename(month, day)

  (1 <= month <= 12) || throw(DomainError(month))
  (1 <= day <= 31) || throw(DomainError(day))

  fn = data_root * @sprintf("/WN_L2_HHD_v8.0_UTM_CALMETWIND_JBT_UTC_2023-%02d-%02d.nc", month, day)
  return fn
end

function load_data(month, day; kwargs...)

  fn = get_filename(month, day)

  return load_data(fn; kwargs...)

end

function join_data(d1::WD, d2::WD) where {WD <: WindData}

  @assert d1.X == d2.X
  @assert d1.Y == d2.Y
  
  T = vcat(d1.T, d2.T)
  WX = cat(d1.WX, d2.WX; dims=3)
  WY = cat(d1.WY, d2.WY; dims=3)

  return WindData(d1.X, d1.Y, T, WX, WY)
end

function join_data(ds::VWD) where {WD <: WindData, VWD <: AbstractVector{WD}}
  
  X = ds[1].X
  Y = ds[1].Y

  for d in ds
    @assert d.X == X
    @assert d.Y == Y
  end

  T = vcat((d.T for d in ds)...)
  WX = cat((d.WX for d in ds)...; dims=3)
  WY = cat((d.WY for d in ds)...; dims=3)

  return WindData(X, Y, T, WX, WY)
end

  

function load_data(fn; starts=[1,1,1], counts=[-1,-1,-1])

  X, Y, T = load_axes(fn; starts=starts, counts=counts)

  W_scale = ncgetatt(fn, "wind_speed_HHM", "scaling_factor")
  W = ncread(fn, "wind_speed_HHM"; start=[1; starts], count=[1; counts]) .* W_scale
  W = dropdims(W, dims=1)
  W = reverse(W; dims=2)

  W_dir_scale = ncgetatt(fn, "wind_dir_HHM", "scaling_factor")
  W_dir = ncread(fn, "wind_dir_HHM"; start=[1; starts], count=[1; counts]) .* W_dir_scale
  W_dir = dropdims(W_dir; dims=1)
  W_dir = reverse(W_dir; dims=2)

  WX = similar(W)
  WY = similar(W)
  for i in CartesianIndices(W)
    WX[i] = W[i] * cosd.(W_dir[i])
    WY[i] = W[i] * sind.(W_dir[i])
  end

  return WindData(X, Y, T, WX, WY)
end

function to_dataframe(fn::F) where {F <: AbstractString}

  to_dataframe(load_data(fn))

end

function to_dataframe(d::D) where {D <: WindData}

  data = []
  for (it, t) in enumerate(d.T)
    for (ix, x) in enumerate(d.X)
      for (iy, y) in enumerate(d.Y)
        push!(data,
          (t=t, x=x, y=y,
            pos=(x, y),
            wx=d.WX[ix, iy, it],
            wy=d.WY[ix, iy, it])
        )
      end
    end
  end

  return DataFrame(data)

end

function t_inds(data::WD, t) where {WD <: WindData}
  insorted(t, data.T) || throw(DomainError(t, "$t not in data.T"))
  return searchsortedfirst(data.T, t)
end

function t_inds(data::WD, tmin, tmax) where {WD <: WindData}
  tmin <= tmax || throw(ArgumentError("tmin !<= tmax"))
  N = length(data.T)
  return max(1, searchsortedlast(data.T, tmin)):min(N, searchsortedfirst(data.T, tmax))
end

function x_inds(data::WD, x) where {WD <: WindData}
  insorted(x, data.X) || throw(DomainError(x, "$x not in data.X"))
  return searchsortedfirst(data.X, x)
end

function x_inds(data::WD, xmin, xmax) where {WD <: WindData}
  xmin <= xmax || throw(ArgumentError("xmin !<= xmax"))
  N = length(data.X)
  return max(1, searchsortedlast(data.X, xmin)):min(N, searchsortedfirst(data.X, xmax))
end

function y_inds(data::WD, y) where {WD <: WindData}
  insorted(y, data.Y) || throw(DomainError(y, "$y not in data.Y"))
  return searchsortedfirst(data.Y, y)
end

function y_inds(data::WD, ymin, ymax) where {WD <: WindData}
  ymin <= ymax || throw(ArgumentError("ymin !<= ymax"))
  N = length(data.Y)
  return max(1, searchsortedlast(data.Y, ymin)):min(N, searchsortedfirst(data.Y, ymax))
end


@recipe function f(d::WD, t_ind::Integer; plottype=:wx) where {WD <: WindData}
  
  t = to_datetime(d.T[t_ind])

  if plottype == :wx
    M = d.WX[:, :, t_ind]
    title = "WX @ $t"
    clabel = "wx (m/s)"
    cmap = :balance
    clims = (-8, 8)
  end

  if plottype == :wy
    M = d.WY[:, :, t_ind]
    title = "WY @ $t"
    clabel = "wy (m/s)"
    cmap = :balance
    clims = (-8, 8)
  end

  if plottype == :mag
    MX = d.WX[:, :, t_ind]
    MY = d.WY[:, :, t_ind]
    M = sqrt.(MX.^2 + MY.^2)
    title = "W_mag @ $t"
    clabel = "||w|| (m/s)"
    cmap = :amp
    clims = (0, 8)
  end
  
  if plottype == :dir
    MX = d.WX[:, :, t_ind]
    MY = d.WY[:, :, t_ind]
    M = atand.(MY,  MX)
    title = "W_dir @ $t"
    clabel = "w_dir (deg)"
    cmap = :twilight
    clims = (-180, 180)
  end

  @series begin

    seriestype --> :heatmap
    xlabel --> "x [km]"
    ylabel --> "y [km]"
    clabel --> clabel
    zlabel --> clabel
    title --> title
    colormap --> cmap
    clims --> clims

    d.X, d.Y, M'
  end

end



# function subinds_time(data, time_min, time_max)

#   inds = time_min .<= data[!, :t] .<= time_max

#   return inds
# end


# function subinds_pos(data, x, y)

#   inds_x = data[!, :x] .== x
#   inds_y = data[!, :y] .== y

#   return inds_x .&& inds_y
# end

# function subinds_pos(data, xmin, xmax, ymin, ymax)

#   inds_x = xmin .<= data[!, :x] .<= xmax
#   inds_y = ymin .<= data[!, :y] .<= ymax

#   return inds_x .&& inds_y
# end


# function wx_data(data)

#   res = unstack(data, :x, :y, :wx)

#   xs = res.x
#   ys = parse.(Float32, names(res)[2:end]) # hacky but works?

#   d = Array(res[:, 2:end])

#   return xs, ys, identity.(d)

# end

# function wy_data(data)

#   res = unstack(data, :x, :y, :wy)

#   xs = res.x
#   ys = parse.(Float32, names(res)[2:end]) # hacky but works?

#   d = Array(res[:, 2:end])

#   return xs, ys, identity.(d)

# end

# function wx_data(data, t)
#   inds = subinds_time(data, t)
#   return wx_data(data[inds, :])
# end

# function wy_data(data, t)
#   inds = subinds_time(data, t)
#   return wy_data(data[inds, :])
# end

function to_datetime(T)

  t0 = DateTime("2023-01-01", dateformat"y-m-d")
  return unix2datetime.(datetime2unix(t0) .+ 60 * T)
end



end
