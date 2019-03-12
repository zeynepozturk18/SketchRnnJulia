using Pkg; for p in ("Knet","PyCall","Plots"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Base.Iterators: flatten, vcat
using Statistics: mean
using Knet: Knet, AutoGrad, param, param0, mat, RNN, relu, Data, adam, progress, nll, zeroone
using PyCall
push!( LOAD_PATH, "./" )
include("utils.jl")
#using utils

### NPZ did not work because while loading the files encoding must be specified
### and NPZ does not have that capability
#data=npzread("/Users/zeynepozturk/dersler/comp541/data/cat.npz")

@pyimport numpy as np

hparams = Dict(
    "data_set"=>["cat"],  # Our dataset.
    "num_steps"=>500,  # Total number of steps of training. Keep large.
    "save_every"=>100,  # Number of batches per checkpoint creation.
    "max_seq_len"=>250,  # Not used. Will be changed by model. [Eliminate?]
    "dec_rnn_size"=>512,  # Size of decoder.
    "dec_model"=>"lstm",  # Decoder: lstm, layer_norm or hyper.
    "enc_rnn_size"=>256,  # Size of encoder.
    "enc_model"=>"lstm",  # Encoder: lstm, layer_norm or hyper.
    "z_size"=>128,  # Size of latent vector z. Recommend 32, 64 or 128.
    "kl_weight"=>0.5,  # KL weight of loss equation. Recommend 0.5 or 1.0.
    "kl_weight_start"=>0.01,  # KL start weight when annealing.
    "kl_tolerance"=>0.2,  # Level of KL loss at which to stop optimizing for KL.
    "batch_size"=>100,  # Minibatch size. Recommend leaving at 100.
    "grad_clip"=>1.0,  # Gradient clipping. Recommend leaving at 1.0.
    "learning_rate"=>0.001,  # Learning rate.
    "decay_rate"=>0.9999,  # Learning rate decay per minibatch.
    "kl_decay_rate"=>0.99995,  # KL annealing decay rate per minibatch.
    "min_learning_rate"=>0.00001,  # Minimum learning rate.
    "use_recurrent_dropout"=>true,  # Dropout with memory loss. Recomended
    "recurrent_dropout_prob"=>0.90,  # Probability of recurrent dropout keep.
    "use_input_dropout"=>false,  # Input dropout. Recommend leaving False.
    "input_dropout_prob"=>0.90,  # Probability of input dropout keep.
    "use_output_dropout"=>false,  # Output droput. Recommend leaving False.
    "output_dropout_prob"=>0.90,  # Probability of output dropout keep.
    "random_scale_factor"=>0.15,  # Random scaling data augmention proportion.
    "augment_stroke_prob"=>0.10,  # Point dropping augmentation proportion.
    "conditional"=>true,  # When False, use unconditional decoder-only model.
    "is_training"=>true  # Is model training? Recommend keeping true.
)
datasets = hparams["data_set"]
#datasets = readlines("classes.txt")
#println(datasets)
filepath = "/Users/zeynepozturk/dersler/comp541/data/"
train_strokes = Float32[]; valid_strokes = Float32[]; test_strokes = Float32[]
train_y = Float32[]; valid_y = Float32[]; test_y = Float32[]
for (idx, dataset) in enumerate(datasets)
    #println("$idx, $dataset")
    global train_strokes, valid_strokes, test_strokes, train_y, valid_y, test_y
    fullpath = string(filepath,dataset,".npz")
    #println(fullpath)
    data=np.load(fullpath, encoding="latin1")
    #println(size(train_strokes))
    if isempty(train_strokes)
        train_strokes=convert(Array{Array{Float32}},get(data, PyObject, "train"))
        #println(size(train_strokes))
        valid_strokes=convert(Array{Array{Float32}},get(data, PyObject, "valid"))
        test_strokes=convert(Array{Array{Float32}},get(data, PyObject, "test"))
        train_y = repeat([idx],size(train_strokes,1))
        valid_y = repeat([idx],size(valid_strokes,1))
        test_y = repeat([idx],size(test_strokes,1))
    else
        train_strokes=vcat(train_strokes,convert(Array{Array{Float32}},get(data, PyObject, "train")))
        #println(size(train_strokes))
        valid_strokes=vcat(valid_strokes,convert(Array{Array{Float32}},get(data, PyObject, "valid")))
        test_strokes=vcat(test_strokes,convert(Array{Array{Float32}},get(data, PyObject, "test")))
        train_y = vcat(train_y,repeat([idx],size(train_strokes,1)))
        valid_y = vcat(valid_y,repeat([idx],size(valid_strokes,1)))
        test_y = vcat(test_y,repeat([idx],size(test_strokes,1)))
    end
end

#println(size(train_strokes[1]))

all_strokes = vcat(train_strokes, valid_strokes, test_strokes)
num_points = 0

for stroke in all_strokes
    global num_points
  num_points += size(stroke,1)
end
avg_len = num_points / size(all_strokes,1)
println("Dataset nb of instances $(size(all_strokes,1)) ($(size(train_strokes,1))
/ $(size(valid_strokes,1)) / $(size(test_strokes,1))) with average length of strokes $avg_len")


# calculate the max strokes we need.
max_seq_len = get_max_len(all_strokes)
# overwrite the hps with this calculation.
hparams["max_seq_len"] = max_seq_len

println("Maximum sequence length is $max_seq_len")
#println("train max len: ", get_max_len(train_strokes) )

train_set = preprocess(DataLoader(train_strokes))

#println("train max len after: ", get_max_len(train_set.strokes) )
#println("train 1 max len after: ", size(train_set.strokes[1]) )
#println("train 7000 max len after: ", size(train_set.strokes[7000]) )

normalizing_scale_factor = calculate_normalizing_scale_factor(train_set)
#println(normalizing_scale_factor)
