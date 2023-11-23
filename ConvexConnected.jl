using Distributed
using HypothesisTests
using EffectSizes
using GLM

addprocs(...)

@everywhere begin
    using CSV
    using Colors
    using Combinatorics
    using DataFrames
    using Distributions
    using MultivariateStats
    using LinearAlgebra
    using Distances
    using Bootstrap
    using Flux
    using StatsBase
    using MLJBase
    using Polyhedra
end

@everywhere coords = CSV.read(".../munsell_rgb.csv", DataFrame) |> Matrix

@everywhere function luv_convert(i)
	c = convert(Luv, RGB(coords[i, :]...))
	return c.l, c.u, c.v
end

@everywhere const luv_coords = [ luv_convert(i) for i in 1:size(coords, 1) ]

#####################
## Learning colors ##
#####################

@everywhere random_constellations = [ convert(Vector{Vector{Float32}}, hcat(collect.(luv_coords[sample(1:1625, 11; replace=false)]))[:]) for _ in 1:1000 ]

@everywhere function _compare_const(c::Vector{Vector{Float32}}, train::Vector{Tuple{Float64, Float64, Float64}}, test::Vector{Tuple{Float64, Float64, Float64}}, epochs::Int)
    labs_train = [ findmin([ Distances.evaluate(Euclidean(), c[i], [train[j]...]) for i in 1:11 ])[2] for j in 1:size(train, 1) ]
    labs_test = [ findmin([ Distances.evaluate(Euclidean(), c[i], [test[j]...]) for i in 1:11 ])[2] for j in 1:size(test, 1) ]
    hc_train = hcat(train, labs_train)
    hc_test = hcat(test, labs_test)
    x_train = collect.(hc_train[:, 1])
    x_train = convert(Vector{Vector{Float32}}, x_train)
    df_train = reduce(hcat, x_train)
    y_train = Flux.onehotbatch(hc_train[:, 2], 1:11)
    x_test = collect.(hc_test[:, 1])
    x_test = convert(Vector{Vector{Float32}}, x_test)
    df_test = reduce(hcat, x_test)
    y_test = Flux.onehotbatch(hc_test[:, 2], 1:11)
    l1 = Dense(3, 9, Flux.relu)
    l2 = Dense(9, 11)
    Flux_nn = Flux.Chain(l1, l2)
    loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
    ps = Flux.params(Flux_nn)
    nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
    cc_train = Float32[] # classification correctness training set
    cc_test = Float32[] # classification correctness test set
    for epoch in 1:epochs
        Flux.train!(loss, ps, nndata, Flux.ADAM())
        ŷ_train = Flux.onecold(Flux_nn(df_train), 1:11)
        push!(cc_train, sum(labs_train .== ŷ_train) / length(train))
        ŷ_test = Flux.onecold(Flux_nn(df_test), 1:11)
        push!(cc_test, sum(labs_test .== ŷ_test) / length(test))
    end
    return cc_train, cc_test
end

@everywhere function _compare_const(train::Vector{Tuple{Float64, Float64, Float64}}, test::Vector{Tuple{Float64, Float64, Float64}}, epochs::Int)
    labs_train = rand(1:11, size(train, 1))
    labs_test = rand(1:11, size(test, 1))
    hc_train = hcat(train, labs_train)
    hc_test = hcat(test, labs_test)
    x_train = collect.(hc_train[:, 1])
    x_train = convert(Vector{Vector{Float32}}, x_train)
    df_train = reduce(hcat, x_train)
    y_train = Flux.onehotbatch(hc_train[:, 2], 1:11)
    x_test = collect.(hc_test[:, 1])
    x_test = convert(Vector{Vector{Float32}}, x_test)
    df_test = reduce(hcat, x_test)
    y_test = Flux.onehotbatch(hc_test[:, 2], 1:11)
    l1 = Dense(3, 9, Flux.relu)
    l2 = Dense(9, 11)
    Flux_nn = Flux.Chain(l1, l2)
    loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
    ps = Flux.params(Flux_nn)
    nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
    cc_train = Float32[] # classification correctness training set
    cc_test = Float32[] # classification correctness test set
    for epoch in 1:epochs
        Flux.train!(loss, ps, nndata, Flux.ADAM())
        ŷ_train = Flux.onecold(Flux_nn(df_train), 1:11)
        push!(cc_train, sum(labs_train .== ŷ_train) / length(train))
        ŷ_test = Flux.onecold(Flux_nn(df_test), 1:11)
        push!(cc_test, sum(labs_test .== ŷ_test) / length(test))
    end
    return cc_train, cc_test
end

@everywhere function compare_const(rc::Vector{Vector{Float32}}, epochs::Int)
    train, test = partition(luv_coords, .8; shuffle=true)
    rnd = _compare_const(rc, train, test, epochs)
    rnd_nonconvex = _compare_const(rc, train, test, epochs, 0)
    return proto, rnd, rnd_nonconvex, rnd_nonconvex_k
end

cc_res = pmap(i->compare_const(random_constellations[i], 1000), 1:length(random_constellations))

bs(x; n=1000) = bootstrap(mean, x, BasicSampling(n))

ptr = reduce(hcat, first.(first.(cc_res)))
ptr_bs = [ Bootstrap.confint(bs(ptr[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]
ptst = reduce(hcat, last.(first.(cc_res)))
ptst_bs = [ Bootstrap.confint(bs(ptst[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]

rtr = reduce(hcat, first.(getindex.(cc_res, 2)))
rtr_bs = [ Bootstrap.confint(bs(rtr[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]
rtst = reduce(hcat, last.(getindex.(cc_res, 2)))
rtst_bs = [ Bootstrap.confint(bs(rtst[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]

rtrnc = reduce(hcat, first.(getindex.(cc_res, 3)))
rtrnc_bs = [ Bootstrap.confint(bs(rtrnc[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]
rtstnc = reduce(hcat, last.(getindex.(cc_res, 3)))
rtstnc_bs = [ Bootstrap.confint(bs(rtstnc[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]

rtrrn = reduce(hcat, first.(last.(cc_res)))
rtrrn_bs = [ Bootstrap.confint(bs(rtrrn[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]
rtstrn = reduce(hcat, last.(last.(cc_res)))
rtstrn_bs = [ Bootstrap.confint(bs(rtstrn[i, :]), BasicConfInt(.95))[1] for i in 1:1000 ]

# function to generate a chain ordered by minimal distance of n randomly selected Munsell chips
@everywhere function rnd_min_dist_chain(n::Int)
    sel = hcat(collect.(luv_coords[sample(1:1625, n; replace=false)]))[:]
    inds = Int[]
    push!(inds, rand(1:length(sel)))
    for j in 1:length(sel) - 1
        cmp = complement(collect(1:length(sel)), inds)
        m = findmin([ Distances.evaluate(Euclidean(), sel[inds[j]], sel[i]) for i in cmp ])[2]
        push!(inds, cmp[m])
    end
    return sel[inds]
end

@everywhere function convex_deviation(df::DataFrame, grps::Vector{Vector{Tuple{Float64, Float64, Float64}}}, g::Int)
    s = Base.stack(grps[g], dims=1)[:, 3:-1:1]
    v = vrep(s)
    p = polyhedron(v)
    dff = filter(:cats => !=(g), df)
    sm = [ in(Point(Vector(dff[i, 3:-1:1])...), hrep(p)) for i in 1:nrow(dff) ] |> sum
    return sm
end

@everywhere function drlt()
    out = zeros(11)
    while minimum(out) < .02
        out .= normalize(rand(Beta(1, 1), 11), 1)
    end
    return out
end

@everywhere function compute_aulc(accuracies::Vector{Float32})
    n = length(accuracies)
    area = sum(0.5 * (accuracies[i] + accuracies[i+1]) for i in 1:(n-1))
    return area
end

@everywhere function compare_connected(df::DataFrame, epochs::Int)
    train, test = partition(df, .8; shuffle=true)
    x_train = Matrix(train[:, 1:3])
    x_train = [ x_train[i, :] for i in 1:size(x_train, 1) ]
    x_train = convert(Vector{Vector{Float32}}, x_train)
    df_train = reduce(hcat, x_train)
    y_train = Flux.onehotbatch(train.cats, 1:11)
    x_test = Matrix(test[:, 1:3])
    x_test = [ x_test[i, :] for i in 1:size(x_test, 1) ]
    x_test = convert(Vector{Vector{Float32}}, x_test)
    df_test = reduce(hcat, x_test)
    y_test = Flux.onehotbatch(test.cats, 1:11)
    l1 = Dense(3, 9, Flux.relu)
    l2 = Dense(9, 11)
    Flux_nn = Flux.Chain(l1, l2)
    loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
    ps = Flux.params(Flux_nn)
    nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
    cc_train = Float32[] # classification correctness training set
    cc_test = Float32[] # classification correctness test set
    for epoch in 1:epochs
        Flux.train!(loss, ps, nndata, Flux.ADAM())
        ŷ_train = Flux.onecold(Flux_nn(df_train), 1:11)
        push!(cc_train, sum(train.cats .== ŷ_train) / nrow(train))
        ŷ_test = Flux.onecold(Flux_nn(df_test), 1:11)
        push!(cc_test, sum(test.cats .== ŷ_test) / nrow(test))
    end
    return cc_train, cc_test
end

@everywhere function run_connected(n::Int, epochs::Int)
    ch = rnd_min_dist_chain(n)
    lbs = [ findmin([ Distances.evaluate(Euclidean(), ch[i], luv_coords[j]) for i in 1:length(ch) ])[2] for j in 1:length(luv_coords) ]
    prt = MLJBase.partition(1:length(ch), drlt()[1:end-1]...)
    groups = [ luv_coords[[ lbs[i] ∈ prt[j] for i in 1:length(lbs) ]] for j in 1:length(prt) ]
    categories = reduce(vcat, [ repeat([i], length(groups[i])) for i in 1:length(groups) ])
    connect_df = DataFrame(reduce(vcat, groups))
    connect_df.cats = categories
    conv_dev = sum([ convex_deviation(connect_df, groups, i) for i in 1:length(groups) ])
    acc_train, acc_test = compare_connected(connect_df, epochs)
    return conv_dev, compute_aulc(acc_train), compute_aulc(acc_test)
end

res = pmap(_->run_connected(110, 100), 1:1000)

reg_df = DataFrame(dev=zscore(first.(res)), train=zscore(getindex.(res, 2)), test=zscore(last.(res)))

lm(@formula(train ~ dev), reg_df)
lm(@formula(test ~ dev ), reg_df)

@everywhere function _compare_connected(train::DataFrame, test::DataFrame, epochs::Int)
    x_train = Matrix(train[:, 1:3])
    x_train = [ x_train[i, :] for i in 1:size(x_train, 1) ]
    x_train = convert(Vector{Vector{Float32}}, x_train)
    df_train = reduce(hcat, x_train)
    y_train = Flux.onehotbatch(train.cats, 1:11)
    x_test = Matrix(test[:, 1:3])
    x_test = [ x_test[i, :] for i in 1:size(x_test, 1) ]
    x_test = convert(Vector{Vector{Float32}}, x_test)
    df_test = reduce(hcat, x_test)
    y_test = Flux.onehotbatch(test.cats, 1:11)
    l1 = Dense(3, 9, Flux.relu)
    l2 = Dense(9, 11)
    Flux_nn = Flux.Chain(l1, l2)
    loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
    ps = Flux.params(Flux_nn)
    nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
    cc_train = Float32[] # classification correctness training set
    cc_test = Float32[] # classification correctness test set
    for epoch in 1:epochs
        Flux.train!(loss, ps, nndata, Flux.ADAM())
        ŷ_train = Flux.onecold(Flux_nn(df_train), 1:11)
        push!(cc_train, sum(train.cats .== ŷ_train) / nrow(train))
        ŷ_test = Flux.onecold(Flux_nn(df_test), 1:11)
        push!(cc_test, sum(test.cats .== ŷ_test) / nrow(test))
    end
    return cc_train, cc_test
end

@everywhere function compare_all(rc::Vector{Vector{Float32}}, epochs::Int)
    ch = rnd_min_dist_chain(110)
    lbs = [ findmin([ Distances.evaluate(Euclidean(), ch[i], luv_coords[j]) for i in 1:length(ch) ])[2] for j in 1:length(luv_coords) ]
    prt = MLJBase.partition(1:length(ch), drlt()[1:end-1]...)
    groups = [ luv_coords[[ lbs[i] ∈ prt[j] for i in 1:length(lbs) ]] for j in 1:length(prt) ]
    categories = reduce(vcat, [ repeat([i], length(groups[i])) for i in 1:length(groups) ])
    df = DataFrame(reduce(vcat, groups))
    df.cats = categories
    train, test = partition(df, .8; shuffle=true)
    train_vec = Tuple.(eachrow(train[!, 1:3]))
    test_vec = Tuple.(eachrow(test[!, 1:3]))
    rnd = _compare_const(rc, train_vec, test_vec, epochs)
    con = _compare_connected(train, test, epochs)
    return rnd, conn
end

ca_res = pmap(i->compare_all(random_constellations[i], 100), 1:length(random_constellations))

rtr = reduce(hcat, first.(first.(ca_res)))
rtr_bs = [ Bootstrap.confint(bs(rtr[i, :]), BasicConfInt(.95))[1] for i in 1:100 ]
rtst = reduce(hcat, last.(first.(ca_res)))
rtst_bs = [ Bootstrap.confint(bs(rtst[i, :]), BasicConfInt(.95))[1] for i in 1:100 ]

rtrc = reduce(hcat, first.(last.(ca_res)))
rtrc_bs = [ Bootstrap.confint(bs(rtrc[i, :]), BasicConfInt(.95))[1] for i in 1:100 ]
rtstc = reduce(hcat, last.(last.(ca_res)))
rtstc_bs = [ Bootstrap.confint(bs(rtstc[i, :]), BasicConfInt(.95))[1] for i in 1:100 ]

[ pvalue(EqualVarianceTTest(rtr[i, :], rtrc[i, :])) for i in 1:100 ]
train_d = [ effectsize(CohenD(rtr[i, :], rtrc[i, :])) for i in 1:100 ]
[ pvalue(EqualVarianceTTest(rtst[i, :], rtstc[i, :])) for i in 1:100 ]
test_d = [ effectsize(CohenD(rtst[i, :], rtstc[i, :])) for i in 1:100 ]

for_bp_train_conv = mapslices(compute_aulc, rtr, dims=1)
for_bp_test_conv = mapslices(compute_aulc, rtst, dims=1)

for_bp_train_conn = mapslices(compute_aulc, rtrc, dims=1)
for_bp_test_conn = mapslices(compute_aulc, rtstc, dims=1)

EqualVarianceTTest(for_bp_train_conv[:], for_bp_train_conn[:])
EqualVarianceTTest(for_bp_test_conv[:], for_bp_test_conn[:])
effectsize(CohenD(for_bp_train_conv[:], for_bp_train_conn[:]))
effectsize(CohenD(for_bp_test_conv[:], for_bp_test_conn[:]))

df_train = DataFrame(epochs=1:100, rnd_res=first.(rtr_bs), ymin2=getindex.(rtr_bs, 2), ymax2=last.(rtr_bs),
                     rndc_res=first.(rtrc_bs), ymin4=getindex.(rtrc_bs, 2), ymax4=last.(rtrc_bs), cd = train_d)
df_test = DataFrame(epochs=1:100, rnd_res=first.(rtst_bs), ymin2=getindex.(rtst_bs, 2), ymax2=last.(rtst_bs),
                    rndc_res=first.(rtstc_bs), ymin4=getindex.(rtstc_bs, 2), ymax4=last.(rtstc_bs), cd = test_d)

#####################
## Learning shapes ##
#####################

@everywhere function create_connected_vase_bowl_regions(n::Int)
    m = reshape(shuffle!(vcat(collect(1:n), fill(0, 49 - n))), (7, 7))
    while any(==(0), m)
        ind = rand(1:n)
        c1, c2 = Tuple(rand(findall(==(ind), m)))
        r = rand([(0, 1), (1, 0), (0, -1), (-1, 0)])
        c3, c4 = minimum([7, maximum([1, c1 + first(r)])]), minimum([7, maximum([1, c2 + last(r)])])
        m[c3, c4] == 0 ? m[c3, c4] = ind : nothing
    end
    return m
end

@everywhere simmeans = Matrix(CSV.read(".../similarity_means.csv"), DataFrame; header=false)

@everywhere mds = MultivariateStats.fit(MDS, simmeans, maxoutdim=3; distances=true)

MultivariateStats.stress(mds)

@everywhere crds = StatsBase.predict(mds)

meshscatter(crds[1, :], crds[2, :], crds[3, :], markersize=.1, color=palette[ccr[:]])

v = vrep(hcat(crds[1, :][ccr[:] .== 2], crds[2, :][ccr[:] .== 2], crds[3, :][ccr[:] .== 2]))
p = polyhedron(v)
m = Polyhedra.Mesh(p)

v1 = vrep(hcat(crds[1, :][ccr[:] .== 1], crds[2, :][ccr[:] .== 1], crds[3, :][ccr[:] .== 1]))
v2 = vrep(hcat(crds[1, :][ccr[:] .== 2], crds[2, :][ccr[:] .== 2], crds[3, :][ccr[:] .== 2]))
p1 = polyhedron(v1)
p2 = polyhedron(v2)
p_int = intersect(p1, p2)
intersect(p1, p2) |> Polyhedra.volume
m1 = Polyhedra.Mesh(p1)
m2 = Polyhedra.Mesh(p2)
m_int = Polyhedra.Mesh(p_int)

@everywhere const coord = Matrix(crds') # coordinates of 49 vase/bowl shapes in 3D Euclidean space

function fit_convex(n::Int, epochs::Int)
	gps = coord[sample(1:49, n, replace=false), :]
	pre_labs = [ findmin([ Distances.evaluate(Euclidean(), gps[i, :], coord[j, :]) for i in 1:n ])[2] for j in 1:49 ]
	train, test = partition(hcat(coord, pre_labs), .8; shuffle=true)
	df_train = reduce(hcat, convert(Vector{Vector{Float32}}, collect(eachrow(train[:, 1:3]))))
	y_train = Flux.onehotbatch(train[:, 4], 1:n)
	df_test = reduce(hcat, convert(Vector{Vector{Float32}}, collect(eachrow(test[:, 1:3]))))
	y_test = Flux.onehotbatch(test[:, 4], 1:n)
	l1 = Dense(3, 9, relu)
    l2 = Dense(9, 9, relu)
	l3 = Dense(9, n)
	Flux_nn = Flux.Chain(l1, Dropout(.33), l2, l3)
	loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
	ps = Flux.params(Flux_nn)
	nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
	cc_train = Float32[] # classification correctness training set
	cc_test = Float32[] # classification correctness test set
	for epoch in 1:epochs
  		Flux.train!(loss, ps, nndata, Flux.ADAM())
  		ŷ_train = Flux.onecold(Flux_nn(df_train), 1:n)
  		push!(cc_train, sum(train[:, 4] .== ŷ_train) / size(train, 1))
  		ŷ_test = Flux.onecold(Flux_nn(df_test), 1:n)
  		push!(cc_test, sum(test[:, 4] .== ŷ_test) / size(test, 1))
	end
	return cc_train, cc_test
end

function fit_connected(n::Int, epochs::Int)
	pre_labs = create_connected_vase_bowl_regions(n)[:]
	train, test = partition(hcat(coord, pre_labs), .8; shuffle=true)
	df_train = reduce(hcat, convert(Vector{Vector{Float32}}, collect(eachrow(train[:, 1:3]))))
	y_train = Flux.onehotbatch(train[:, 4], 1:n)
	df_test = reduce(hcat, convert(Vector{Vector{Float32}}, collect(eachrow(test[:, 1:3]))))
	y_test = Flux.onehotbatch(test[:, 4], 1:n)
	l1 = Dense(3, 9, relu)
    l2 = Dense(9, 9, relu)
	l3 = Dense(9, n)
	Flux_nn = Flux.Chain(l1, Dropout(.33), l2, l3)
	loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
	ps = Flux.params(Flux_nn)
	nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
	cc_train = Float32[] # classification correctness training set
	cc_test = Float32[] # classification correctness test set
	for epoch in 1:epochs
  		Flux.train!(loss, ps, nndata, Flux.ADAM())
  		ŷ_train = Flux.onecold(Flux_nn(df_train), 1:n)
  		push!(cc_train, sum(train[:, 4] .== ŷ_train) / size(train, 1))
  		ŷ_test = Flux.onecold(Flux_nn(df_test), 1:n)
  		push!(cc_test, sum(test[:, 4] .== ŷ_test) / size(test, 1))
	end
	return cc_train, cc_test
end

bs(x; n=1000) = bootstrap(mean, x, BasicSampling(n))

# convex
cnvx_res2 = [ fit_convex(2, 100) for _ in 1:1000 ]

cn_train2 = reduce(hcat, [ first(cnvx_res2[i]) for i in 1:length(cnvx_res2) ])
cntr_bs2 = [ Bootstrap.confint(bs(cn_train2[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_train2, 1) ]
cn_test2 = reduce(hcat, [ last(cnvx_res2[i]) for i in 1:length(cnvx_res2) ])
cnts_bs2 = [ Bootstrap.confint(bs(cn_test2[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_test2, 1) ]

catr2 = [ compute_aulc(cn_train2[:, i]) for i in 1:length(cnvx_res2) ]
b2 = Bootstrap.confint(bs(catr2), BasicConfInt(.95))[1]
cats2 = [ compute_aulc(cn_test2[:, i]) for i in 1:length(cnvx_res2) ]
bs2 = Bootstrap.confint(bs(cats2), BasicConfInt(.95))[1]

epochs = 1:size(cn_train2, 1)
df_tt2 = DataFrame(epochs=epochs, restr=first.(cntr_bs2), ymin1=getindex.(cntr_bs2, 2), ymax1=last.(cntr_bs2), rests=first.(cnts_bs2), ymin2=getindex.(cnts_bs2, 2), ymax2=last.(cnts_bs2))

cnvx_res3 = [ fit_convex(3, 100) for _ in 1:1000 ]

cn_train3 = reduce(hcat, [ first(cnvx_res3[i]) for i in 1:length(cnvx_res3) ])
cntr_bs3 = [ Bootstrap.confint(bs(cn_train3[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_train3, 1) ]
cn_test3 = reduce(hcat, [ last(cnvx_res3[i]) for i in 1:length(cnvx_res3) ])
cnts_bs3 = [ Bootstrap.confint(bs(cn_test3[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_test3, 1) ]

catr3 = [ compute_aulc(cn_train3[:, i]) for i in 1:length(cnvx_res3) ]
b3 = Bootstrap.confint(bs(catr3), BasicConfInt(.95))[1]
cats3 = [ compute_aulc(cn_test3[:, i]) for i in 1:length(cnvx_res3) ]
bs3 = Bootstrap.confint(bs(cats3), BasicConfInt(.95))[1]

epochs = 1:size(cn_train3, 1)
df_tt3 = DataFrame(epochs=epochs, restr=first.(cntr_bs3), ymin1=getindex.(cntr_bs3, 2), ymax1=last.(cntr_bs3), rests=first.(cnts_bs3), ymin2=getindex.(cnts_bs3, 2), ymax2=last.(cnts_bs3))

cnvx_res4 = [ fit_convex(4, 100) for _ in 1:1000 ]

cn_train4 = reduce(hcat, [ first(cnvx_res4[i]) for i in 1:length(cnvx_res4) ])
cntr_bs4 = [ Bootstrap.confint(bs(cn_train4[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_train4, 1) ]
cn_test4 = reduce(hcat, [ last(cnvx_res4[i]) for i in 1:length(cnvx_res4) ])
cnts_bs4 = [ Bootstrap.confint(bs(cn_test4[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_test4, 1) ]

catr4 = [ compute_aulc(cn_train4[:, i]) for i in 1:length(cnvx_res4) ]
b4 = Bootstrap.confint(bs(catr4), BasicConfInt(.95))[1]
cats4 = [ compute_aulc(cn_test4[:, i]) for i in 1:length(cnvx_res4) ]
bs4 = Bootstrap.confint(bs(cats4), BasicConfInt(.95))[1]

epochs = 1:size(cn_train4, 1)
df_tt4 = DataFrame(epochs=epochs, restr=first.(cntr_bs4), ymin1=getindex.(cntr_bs4, 2), ymax1=last.(cntr_bs4), rests=first.(cnts_bs4), ymin2=getindex.(cnts_bs4, 2), ymax2=last.(cnts_bs4))

cnvx_res5 = [ fit_convex(5, 100) for _ in 1:1000 ]

cn_train5 = reduce(hcat, [ first(cnvx_res5[i]) for i in 1:length(cnvx_res5) ])
cntr_bs5 = [ Bootstrap.confint(bs(cn_train5[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_train5, 1) ]
cn_test5 = reduce(hcat, [ last(cnvx_res5[i]) for i in 1:length(cnvx_res5) ])
cnts_bs5 = [ Bootstrap.confint(bs(cn_test5[i, :]), BasicConfInt(.95))[1] for i in 1:size(cn_test5, 1) ]

catr5 = [ compute_aulc(cn_train5[:, i]) for i in 1:length(cnvx_res5) ]
b5 = Bootstrap.confint(bs(catr5), BasicConfInt(.95))[1]
cats5 = [ compute_aulc(cn_test5[:, i]) for i in 1:length(cnvx_res5) ]
bs5 = Bootstrap.confint(bs(cats5), BasicConfInt(.95))[1]

epochs = 1:size(cn_train5, 1)
df_tt5 = DataFrame(epochs=epochs, restr=first.(cntr_bs5), ymin1=getindex.(cntr_bs5, 2), ymax1=last.(cntr_bs5), rests=first.(cnts_bs5), ymin2=getindex.(cnts_bs5, 2), ymax2=last.(cnts_bs5))

# connected

cnc_res2 = [ fit_connected(2, 100) for _ in 1:1000 ]

cnc_train2 = reduce(hcat, [ first(cnc_res2[i]) for i in 1:length(cnc_res2) ])
cnctr_bs2 = [ Bootstrap.confint(bs(cnc_train2[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_train2, 1) ]
cnc_test2 = reduce(hcat, [ last(cnc_res2[i]) for i in 1:length(cnc_res2) ])
cncts_bs2 = [ Bootstrap.confint(bs(cnc_test2[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_test2, 1) ]

cncatr2 = [ compute_aulc(cnc_train2[:, i]) for i in 1:length(cnc_res2) ]
cnb2 = Bootstrap.confint(bs(cncatr2), BasicConfInt(.95))[1]
cncats2 = [ compute_aulc(cnc_test2[:, i]) for i in 1:length(cnc_res2) ]
cnbs2 = Bootstrap.confint(bs(cncats2), BasicConfInt(.95))[1]

dfn_tt2 = DataFrame(epochs=epochs, restr=first.(cnctr_bs2), ymin1=getindex.(cnctr_bs2, 2), ymax1=last.(cnctr_bs2), rests=first.(cncts_bs2), ymin2=getindex.(cncts_bs2, 2), ymax2=last.(cncts_bs2))

cnc_res3 = [ fit_connected(3, 100) for _ in 1:1000 ]

cnc_train3 = reduce(hcat, [ first(cnc_res3[i]) for i in 1:length(cnc_res3) ])
cnctr_bs3 = [ Bootstrap.confint(bs(cnc_train3[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_train3, 1) ]
cnc_test3 = reduce(hcat, [ last(cnc_res3[i]) for i in 1:length(cnc_res3) ])
cncts_bs3 = [ Bootstrap.confint(bs(cnc_test3[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_test3, 1) ]

cncatr3 = [ compute_aulc(cnc_train3[:, i]) for i in 1:length(cnc_res3) ]
cnb3 = Bootstrap.confint(bs(cncatr3), BasicConfInt(.95))[1]
cncats3 = [ compute_aulc(cnc_test3[:, i]) for i in 1:length(cnc_res3) ]
cnbs3 = Bootstrap.confint(bs(cncats3), BasicConfInt(.95))[1]

dfn_tt3 = DataFrame(epochs=epochs, restr=first.(cnctr_bs3), ymin1=getindex.(cnctr_bs3, 2), ymax1=last.(cnctr_bs3), rests=first.(cncts_bs3), ymin2=getindex.(cncts_bs3, 2), ymax2=last.(cncts_bs3))

cnc_res4 = [ fit_connected(4, 100) for _ in 1:1000 ]

cnc_train4 = reduce(hcat, [ first(cnc_res4[i]) for i in 1:length(cnc_res4) ])
cnctr_bs4 = [ Bootstrap.confint(bs(cnc_train4[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_train4, 1) ]
cnc_test4 = reduce(hcat, [ last(cnc_res4[i]) for i in 1:length(cnc_res4) ])
cncts_bs4 = [ Bootstrap.confint(bs(cnc_test4[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_test4, 1) ]

cncatr4 = [ compute_aulc(cnc_train4[:, i]) for i in 1:length(cnc_res4) ]
cnb4 = Bootstrap.confint(bs(cncatr4), BasicConfInt(.95))[1]
cncats4 = [ compute_aulc(cnc_test4[:, i]) for i in 1:length(cnc_res4) ]
cnbs4 = Bootstrap.confint(bs(cncats4), BasicConfInt(.95))[1]

dfn_tt4 = DataFrame(epochs=epochs, restr=first.(cnctr_bs4), ymin1=getindex.(cnctr_bs4, 2), ymax1=last.(cnctr_bs4), rests=first.(cncts_bs4), ymin2=getindex.(cncts_bs4, 2), ymax2=last.(cncts_bs4))

cnc_res5 = [ fit_connected(5, 100) for _ in 1:1000 ]

cnc_train5 = reduce(hcat, [ first(cnc_res5[i]) for i in 1:length(cnc_res5) ])
cnctr_bs5 = [ Bootstrap.confint(bs(cnc_train5[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_train5, 1) ]
cnc_test5 = reduce(hcat, [ last(cnc_res5[i]) for i in 1:length(cnc_res5) ])
cncts_bs5 = [ Bootstrap.confint(bs(cnc_test5[i, :]), BasicConfInt(.95))[1] for i in 1:size(cnc_test5, 1) ]

cncatr5 = [ compute_aulc(cnc_train5[:, i]) for i in 1:length(cnc_res5) ]
cnb5 = Bootstrap.confint(bs(cncatr5), BasicConfInt(.95))[1]
cncats5 = [ compute_aulc(cnc_test5[:, i]) for i in 1:length(cnc_res5) ]
cnbs5 = Bootstrap.confint(bs(cncats5), BasicConfInt(.95))[1]

dfn_tt5 = DataFrame(epochs=epochs, restr=first.(cnctr_bs5), ymin1=getindex.(cnctr_bs5, 2), ymax1=last.(cnctr_bs5), rests=first.(cncts_bs5), ymin2=getindex.(cncts_bs5, 2), ymax2=last.(cncts_bs5))

bxtrcv = vcat(catr2, catr3, catr4, catr5)
bxtrcn = vcat(cncatr2, cncatr3, cncatr4, cncatr5)
bxtscv = vcat(cats2, cats3, cats4, cats5)
bxtscn = vcat(cncats2, cncats3, cncats4, cncats5)

EqualVarianceTTest([ compute_aulc(cn_train2[:, i]) for i in 1:length(cnvx_res2) ], [ compute_aulc(cnc_train2[:, i]) for i in 1:length(cnc_res2) ])
EqualVarianceTTest([ compute_aulc(cn_train3[:, i]) for i in 1:length(cnvx_res3) ], [ compute_aulc(cnc_train3[:, i]) for i in 1:length(cnc_res3) ])
EqualVarianceTTest([ compute_aulc(cn_train4[:, i]) for i in 1:length(cnvx_res4) ], [ compute_aulc(cnc_train4[:, i]) for i in 1:length(cnc_res4) ])
EqualVarianceTTest([ compute_aulc(cn_train5[:, i]) for i in 1:length(cnvx_res5) ], [ compute_aulc(cnc_train5[:, i]) for i in 1:length(cnc_res5) ])

effectsize(CohenD([ compute_aulc(cn_train2[:, i]) for i in 1:length(cnvx_res2) ], [ compute_aulc(cnc_train2[:, i]) for i in 1:length(cnc_res2) ]))
effectsize(CohenD([ compute_aulc(cn_train3[:, i]) for i in 1:length(cnvx_res3) ], [ compute_aulc(cnc_train3[:, i]) for i in 1:length(cnc_res3) ]))
effectsize(CohenD([ compute_aulc(cn_train4[:, i]) for i in 1:length(cnvx_res4) ], [ compute_aulc(cnc_train4[:, i]) for i in 1:length(cnc_res4) ]))
effectsize(CohenD([ compute_aulc(cn_train5[:, i]) for i in 1:length(cnvx_res5) ], [ compute_aulc(cnc_train5[:, i]) for i in 1:length(cnc_res5) ]))

EqualVarianceTTest([ compute_aulc(cn_test2[:, i]) for i in 1:length(cnvx_res2) ], [ compute_aulc(cnc_test2[:, i]) for i in 1:length(cnc_res2) ])
EqualVarianceTTest([ compute_aulc(cn_test3[:, i]) for i in 1:length(cnvx_res3) ], [ compute_aulc(cnc_test3[:, i]) for i in 1:length(cnc_res3) ])
EqualVarianceTTest([ compute_aulc(cn_test4[:, i]) for i in 1:length(cnvx_res4) ], [ compute_aulc(cnc_test4[:, i]) for i in 1:length(cnc_res4) ])
EqualVarianceTTest([ compute_aulc(cn_test5[:, i]) for i in 1:length(cnvx_res5) ], [ compute_aulc(cnc_test5[:, i]) for i in 1:length(cnc_res5) ])

effectsize(CohenD([ compute_aulc(cn_test2[:, i]) for i in 1:length(cnvx_res2) ], [ compute_aulc(cnc_test2[:, i]) for i in 1:length(cnc_res2) ]))
effectsize(CohenD([ compute_aulc(cn_test3[:, i]) for i in 1:length(cnvx_res3) ], [ compute_aulc(cnc_test3[:, i]) for i in 1:length(cnc_res3) ]))
effectsize(CohenD([ compute_aulc(cn_test4[:, i]) for i in 1:length(cnvx_res4) ], [ compute_aulc(cnc_test4[:, i]) for i in 1:length(cnc_res4) ]))
effectsize(CohenD([ compute_aulc(cn_test5[:, i]) for i in 1:length(cnvx_res5) ], [ compute_aulc(cnc_test5[:, i]) for i in 1:length(cnc_res5) ]))

# calculate convexity deviation for connected regions and correlate with aulc

@everywhere function connected_regress(n::Int, epochs::Int)
    ccr = create_connected_vase_bowl_regions(n)
    v = [ vrep(hcat(crds[1, :][ccr[:] .== i], crds[2, :][ccr[:] .== i], crds[3, :][ccr[:] .== i])) for i in 1:length(unique(ccr)) ]
    p = [ polyhedron(v[i]) for i in 1:length(v) ]
    p_int = [ intersect(p[first(i)], p[last(i)]) for i in  collect(combinations(1:n, 2)) ]
    dev = sum(Polyhedra.volume.(p_int))
	pre_labs = ccr[:]
	train, test = partition(hcat(coord, pre_labs), .8; shuffle=true)
	df_train = reduce(hcat, convert(Vector{Vector{Float32}}, collect(eachrow(train[:, 1:3]))))
	y_train = Flux.onehotbatch(train[:, 4], 1:n)
	df_test = reduce(hcat, convert(Vector{Vector{Float32}}, collect(eachrow(test[:, 1:3]))))
	y_test = Flux.onehotbatch(test[:, 4], 1:n)
	l1 = Dense(3, 9, relu)
    l2 = Dense(9, 9, relu)
	l3 = Dense(9, n)
	Flux_nn = Flux.Chain(l1, Dropout(.33), l2, l3)
	loss(x, y) = Flux.logitcrossentropy(Flux_nn(x), y)
	ps = Flux.params(Flux_nn)
	nndata = Flux.Data.DataLoader((df_train, y_train), batchsize=3, shuffle=true)
	cc_train = Float32[] # classification correctness training set
	cc_test = Float32[] # classification correctness test set
	for epoch in 1:epochs
  		Flux.train!(loss, ps, nndata, Flux.ADAM())
  		ŷ_train = Flux.onecold(Flux_nn(df_train), 1:n)
  		push!(cc_train, sum(train[:, 4] .== ŷ_train) / size(train, 1))
  		ŷ_test = Flux.onecold(Flux_nn(df_test), 1:n)
  		push!(cc_test, sum(test[:, 4] .== ŷ_test) / size(test, 1))
	end
	return dev, compute_aulc(cc_train), compute_aulc(cc_test)
end

reg2 = pmap(_->connected_regress(2, 100), 1:1000)
reg2_df = DataFrame(dev=zscore(first.(reg2)), train=zscore(getindex.(reg2, 2)), test=zscore(last.(reg2)))
lm(@formula(train ~ dev), reg2_df)
lm(@formula(test ~ dev), reg2_df)

reg3 = pmap(_->connected_regress(3, 100), 1:1000)
reg3_df = DataFrame(dev=zscore(first.(reg3)), train=zscore(getindex.(reg3, 2)), test=zscore(last.(reg3)))
lm(@formula(train ~ dev), reg3_df)
lm(@formula(test ~ dev), reg3_df)

reg4 = pmap(_->connected_regress(4, 100), 1:1000)
reg4_df = DataFrame(dev=zscore(first.(reg4)), train=zscore(getindex.(reg4, 2)), test=zscore(last.(reg4)))
lm(@formula(train ~ dev), reg4_df)
lm(@formula(test ~ dev), reg4_df)

reg5 = pmap(_->connected_regress(5, 100), 1:1000)
reg5_df = DataFrame(dev=zscore(first.(reg5)), train=zscore(getindex.(reg5, 2)), test=zscore(last.(reg5)))
lm(@formula(train ~ dev), reg5_df)
lm(@formula(test ~ dev), reg5_df)
