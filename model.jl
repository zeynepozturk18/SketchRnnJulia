using Knet

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
