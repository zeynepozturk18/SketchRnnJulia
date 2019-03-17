using Pkg; for p in ("Knet","Distributions"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Knet
using Distributions

function copy_hparams(hparams)
  """Return a copy of an HParams instance."""
  copy_params=deepcopy(hparams)
  return copy_params
end

function get_default_hparams()
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
    return hparams
end

function sample(model, seq_len=250, temperature=1.0, greedy_mode=false, z=nothing)
  """Samples a sequence from a pre-trained model."""
  ## for the baseline samples randomly

  function adjust_temp(pi_pdf, temp)
    pi_pdf = log.(pi_pdf) / temp
    pi_pdf .-= max(pi_pdf)
    pi_pdf = exp.(pi_pdf)
    pi_pdf ./= sum(pi_pdf)
    return pi_pdf
  end

  function get_pi_idx(x, pdf, temp=1.0, greedy=False)
    """Samples from a pdf, optionally greedily."""
    if greedy
      return findmax(pdf)[2]
    end
    pdf = adjust_temp(deepcopy(pdf), temp)
    accumulate = 0
    for i in 1:size(pdf,1)
      accumulate += pdf[i]
      if accumulate >= x
        return i
      end
    println("Error with sampling ensemble.")
    return -1
    end
  end

  function sample_gaussian_2d(mu1, mu2, s1, s2, rho, temp=1.0, greedy=false)
    if greedy
      return mu1, mu2
    end
    mean = [mu1, mu2]
    s1 *= temp * temp
    s2 *= temp * temp
    cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
    x = rand(MvNormal(mean, cov),1)
    return x[1], x[2]
  end

  prev_x = zeros(Float32, 1, 1, 5)
  prev_x[0, 0, 2] = 1  # initially, we want to see beginning of new stroke
  if !@isdefined z
    z = randn(1, model.hps["z_size"])  # not used if unconditional
  end

  #=
  if  !model.hps["conditional"]
    prev_state =
  else
    prev_state =
  end
  =#

  strokes = zeros(Float32, seq_len, 5)
  mixture_params = []

  greedy = greedy_mode
  temp = temperature

  for i in 1:seq_len
    #=
    if !model.hps["conditional"]
      feed =
    else
      feed =
    end

    #use pretrained model
    params = [
        model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
        model.pen, model.final_state
    ])
    =#

    [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = params

    idx = get_pi_idx(rand(), o_pi[1], temp, greedy)

    idx_eos = get_pi_idx(rand(), o_pen[1], temp, greedy)
    eos = [0, 0, 0]
    eos[idx_eos] = 1

    next_x1, next_x2 = sample_gaussian_2d(o_mu1[1][idx], o_mu2[1][idx],
                                          o_sigma1[1][idx], o_sigma2[1][idx],
                                          o_corr[1][idx], sqrt(temp), greedy)

    strokes[i, :] = [next_x1, next_x2, eos[1], eos[2], eos[3]]

    params = [
        o_pi[1], o_mu1[1], o_mu2[1], o_sigma1[1], o_sigma2[1], o_corr[1],
        o_pen[1]
        ]

    push!(mixture_params, params)

    prev_x = zeros(Float32, 1, 1, 5)
    prev_x[1][1] = [next_x1, next_x2, eos[1], eos[2], eos[3]]
    prev_state = next_state

  end
  return strokes, mixture_params
end
