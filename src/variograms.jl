module Variograms

using LinearAlgebra, LsqFit, StatsBase, ProgressMeter


function empirical_variogram_spatiotemporal(times, positions, values, trange, prange)

    # check sizes
    length(times) == size(positions, 1) == length(values) || throw(DimensionMismatch())

    N = length(values) # number of datapoints
    Nt = length(trange)
    Np = length(prange)

    # 2d matrices to store counts and sums
    counts = zeros(Nt, Np) 
    totals = zeros(Nt, Np)

    @showprogress for i=1:N, j=(i+1):N # for every pair of datapoints
        dt = abs(times[j] - times[i])
        if dt <= trange[end] + step(trange)
            dp = norm(positions[j, :] - positions[i, :])
            if  dp <= prange[end] + step(prange)
                t_ind = searchsortedlast(trange, dt)
                p_ind = searchsortedlast(prange, dp)
                counts[t_ind, p_ind] += 1
                totals[t_ind, p_ind] += (values[j] - values[i])^2
            end
        end
    end

    return totals ./ (2 * counts), counts
end







function empirical_variogram(pos, v, drange; N=1000)

    size(pos, 1) == length(v) || throw(DimensionMismatch())
    # expects pos to be NxM matrix, where M is the dimension of the "pos" variable
    # expects v to be N vector

    N_data = size(pos, 1)
    if N_data <= N
        inds = 1:N_data
    else
        inds = rand(1:N_data, N)
    end

    pos_ = pos[inds, :]
    v_ = v[inds]

    L = length(v_)
    D = length(drange)
    counts = zeros(D)
    vs = zeros(D)

    for i=1:L, j=i:L
        d = norm(pos_[i, :] - pos_[j, :])
        
        if d <= 1.1*drange[end]
            
            ind = searchsortedlast(drange, d)
            
            counts[ind] += 1
            vs[ind] += (v_[i] - v_[j])^2
            
        end
    end

    return vs ./ (2 * counts), counts
end


function variogram_SE(x, p)
    σ, l = p
    return σ * (1 .- exp.( - 1/2 * (x/l).^2 ) )
end

function variogram_Matern_12(x, p)
    σ, l = p
    return σ * (1 .- exp.( - (x/l) ) )
end

function fit_variogram_SE(ds, γs, p0 = [1.0, 1.0])
    fit = curve_fit(variogram_SE, ds, γs, p0)
    return fit.param
end

function fit_variogram_Matern_12(ds, γs, p0 = [1.0, 1.0])
    fit = curve_fit(variogram_Matern_12, ds, γs, p0)
    return fit.param
end


function estimate_σt(ts, ws)

    vs = Float64[]
    for i=1:(length(ts)-1), j=(i+1):length(ts)

        dt = ts[j]-ts[i]
        dw = ws[j] - ws[i]

        push!(vs, dw / sqrt(dt))
    end

    return std(vs)
end

end