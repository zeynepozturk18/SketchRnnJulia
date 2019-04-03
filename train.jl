using Pkg; for p in ("Knet","PyCall","Plots"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Base.Iterators: flatten, vcat
using Statistics: mean
using Knet: Knet, AutoGrad, param, param0, mat, RNN, relu, Data, adam, progress, nll, zeroone
using PyCall
push!( LOAD_PATH, "./" )
include("utils.jl")
#include("model.jl")
include("baseline.jl")

#using utils

### NPZ did not work because while loading the files encoding must be specified
### and NPZ does not have that capability
#data=npzread("/Users/zeynepozturk/dersler/comp541/data/cat.npz")

#@pyimport numpy as np
np=pyimport("numpy")

# data_dir = "/Users/zeynepozturk/dersler/comp541/data/" #mac
data_dir = "/home/zeynep/Downloads/comp541/SketchRNN/data/" #ubuntu

model_params=get_default_hparams()

function load_dataset(data_dir, model_params; inference_mode=false)
    train_strokes = Float32[]; valid_strokes = Float32[]; test_strokes = Float32[]
    #train_y = Float32[]; valid_y = Float32[]; test_y = Float32[]
    datasets = model_params["data_set"]
    #datasets = readlines("classes.txt")
    #println(datasets)
    for (idx, dataset) in enumerate(datasets)
        #println("$idx, $dataset")
        #global train_strokes, valid_strokes, test_strokes#, train_y, valid_y, test_y
        fullpath = string(data_dir,dataset,".npz")
        #println(fullpath)
        data=np.load(fullpath, encoding="latin1")
        #println(size(train_strokes))
        if isempty(train_strokes)
            train_strokes=convert(Array{Array{Float32}},get(data, PyObject, "train"))
            #println(size(train_strokes))
            valid_strokes=convert(Array{Array{Float32}},get(data, PyObject, "valid"))
            test_strokes=convert(Array{Array{Float32}},get(data, PyObject, "test"))
            #train_y = repeat([idx],size(train_strokes,1))
            #valid_y = repeat([idx],size(valid_strokes,1))
            #test_y = repeat([idx],size(test_strokes,1))
        else
            train_strokes=vcat(train_strokes,convert(Array{Array{Float32}},get(data, PyObject, "train")))
            #println(size(train_strokes))
            valid_strokes=vcat(valid_strokes,convert(Array{Array{Float32}},get(data, PyObject, "valid")))
            test_strokes=vcat(test_strokes,convert(Array{Array{Float32}},get(data, PyObject, "test")))
            #train_y = vcat(train_y,repeat([idx],size(train_strokes,1)))
            #valid_y = vcat(valid_y,repeat([idx],size(valid_strokes,1)))
            #test_y = vcat(test_y,repeat([idx],size(test_strokes,1)))
        end
    end

    #println(size(train_strokes[1]))

    all_strokes = vcat(train_strokes, valid_strokes, test_strokes)
    num_points = 0

    for stroke in all_strokes
        #global num_points
      num_points += size(stroke,1)
    end
    avg_len = num_points / size(all_strokes,1)
    println("Dataset nb of instances $(size(all_strokes,1)) ($(size(train_strokes,1))
    / $(size(valid_strokes,1)) / $(size(test_strokes,1))) with average length of strokes $avg_len")


    # calculate the max strokes we need.
    max_seq_len = get_max_len(all_strokes)
    # overwrite the hps with this calculation.
    model_params["max_seq_len"] = max_seq_len

    println("Maximum sequence length is $max_seq_len")
    #println("train max len: ", get_max_len(train_strokes) )

    eval_model_params = copy_hparams(model_params)

    eval_model_params["use_input_dropout"] = 0
    eval_model_params["use_recurrent_dropout"] = 0
    eval_model_params["use_output_dropout"] = 0
    eval_model_params["is_training"] = 1

    if inference_mode
      eval_model_params["batch_size"] = 1
      eval_model_params["is_training"] = 0
    end

    sample_model_params = copy_hparams(eval_model_params)
    sample_model_params["batch_size"] = 1  # only sample one at a time
    sample_model_params["max_seq_len"] = 1  # sample one point at a time

    train_set = preprocess(DataLoader(
        train_strokes;
        batch_size=model_params["batch_size"],
        max_seq_len=model_params["max_seq_len"],
        random_scale_factor=model_params["random_scale_factor"],
        augment_stroke_prob=model_params["augment_stroke_prob"]))

    #println("train max len after: ", get_max_len(train_set.strokes) )
    #println("train 1 max len after: ", size(train_set.strokes[1]) )
    #println("train 7000 max len after: ", size(train_set.strokes[7000]) )

    normalizing_scale_factor = calculate_normalizing_scale_factor(train_set)
    #println(normalizing_scale_factor)

    train_set = normalize(train_set, normalizing_scale_factor)

    valid_set = preprocess(DataLoader(
        valid_strokes;
        batch_size=eval_model_params["batch_size"],
        max_seq_len=eval_model_params["max_seq_len"],
        random_scale_factor=0.0,
        augment_stroke_prob=0.0))
    valid_set = normalize(valid_set, normalizing_scale_factor)

    test_set = preprocess(DataLoader(
        test_strokes;
        batch_size=eval_model_params["batch_size"],
        max_seq_len=eval_model_params["max_seq_len"],
        random_scale_factor=0.0,
        augment_stroke_prob=0.0))
    test_set = normalize(test_set, normalizing_scale_factor)

    result = [
        train_set, valid_set, test_set, model_params, eval_model_params,
        sample_model_params
    ]
    return result
end

train_set, valid_set, test_set, model_params, eval_model_params, sample_model_params = load_dataset(data_dir, model_params; inference_mode=false)

#random_batch(train_set) # one batch at a time as in the paper

# minibatching
_, batches, seq_len = minibatch2(train_set)
println(summary(batches))

#output
#sample(model, seq_len=250, temperature=1.0, greedy_mode=false, z=nothing)

# build_model
hps = get_default_hparams()
kl_cost = build_model(hps) # baseline kl cost
println("baseline kl cost: $kl_cost")

output = baseline_sample(0.01, 1.01)
