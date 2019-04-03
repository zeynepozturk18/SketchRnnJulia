using Pkg; for p in ("Knet","Distributions"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Knet
using Distributions
using Statistics


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
        "num_mixture"=>20,  # Number of mixtures in Gaussian mixture model.
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
#=
class Model(object):
  """Define a SketchRNN model."""

  def __init__(self, hps, gpu_mode=True, reuse=False):
    """Initializer for the SketchRNN model.

    Args:
       hps: a HParams object containing model hyperparameters
       gpu_mode: a boolean that when True, uses GPU mode.
       reuse: a boolean that when true, attemps to reuse variables.
    """
    self.hps = hps
    with tf.variable_scope('vector_rnn', reuse=reuse):
      if not gpu_mode:
        with tf.device('/cpu:0'):
          tf.logging.info('Model using cpu.')
          self.build_model(hps)
      else:
        tf.logging.info('Model using gpu.')
        self.build_model(hps)

  def encoder(self, batch, sequence_lengths):
    """Define the bi-directional encoder module of sketch-rnn."""
    unused_outputs, last_states = tf.nn.bidirectional_dynamic_rnn(
        self.enc_cell_fw,
        self.enc_cell_bw,
        batch,
        sequence_length=sequence_lengths,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='ENC_RNN')

    last_state_fw, last_state_bw = last_states
    last_h_fw = self.enc_cell_fw.get_output(last_state_fw)
    last_h_bw = self.enc_cell_bw.get_output(last_state_bw)
    last_h = tf.concat([last_h_fw, last_h_bw], 1)
    mu = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_mu',
        init_w='gaussian',
        weight_start=0.001)
    presig = rnn.super_linear(
        last_h,
        self.hps.z_size,
        input_size=self.hps.enc_rnn_size * 2,  # bi-dir, so x2
        scope='ENC_RNN_sigma',
        init_w='gaussian',
        weight_start=0.001)
    return mu, presig
=#
function build_model(hps)
    #=
    """Define model architecture."""
    if hps.is_training:
      self.global_step = tf.Variable(0, name='global_step', trainable=False)

    if hps.dec_model == 'lstm':
      cell_fn = rnn.LSTMCell
    elif hps.dec_model == 'layer_norm':
      cell_fn = rnn.LayerNormLSTMCell
    elif hps.dec_model == 'hyper':
      cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'

    if hps.enc_model == 'lstm':
      enc_cell_fn = rnn.LSTMCell
    elif hps.enc_model == 'layer_norm':
      enc_cell_fn = rnn.LayerNormLSTMCell
    elif hps.enc_model == 'hyper':
      enc_cell_fn = rnn.HyperLSTMCell
    else:
      assert False, 'please choose a respectable cell'
    =#
    use_recurrent_dropout = hps["use_recurrent_dropout"]
    use_input_dropout = hps["use_input_dropout"]
    use_output_dropout = hps["use_output_dropout"]

    #=
    cell = cell_fn(
        hps.dec_rnn_size,
        use_recurrent_dropout=use_recurrent_dropout,
        dropout_keep_prob=self.hps.recurrent_dropout_prob)

    if hps.conditional:  # vae mode:
      if hps.enc_model == 'hyper':
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
      else:
        self.enc_cell_fw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
        self.enc_cell_bw = enc_cell_fn(
            hps.enc_rnn_size,
            use_recurrent_dropout=use_recurrent_dropout,
            dropout_keep_prob=self.hps.recurrent_dropout_prob)
    =#

    # dropout:
    println("Input dropout mode = $use_input_dropout.")
    println("Output dropout mode = $use_output_dropout")
    println("Recurrent dropout mode = $use_recurrent_dropout")

    #=
    if use_input_dropout:
      tf.logging.info('Dropout to input w/ keep_prob = %4.4f.',
                      self.hps.input_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, input_keep_prob=self.hps.input_dropout_prob)
    if use_output_dropout:
      tf.logging.info('Dropout to output w/ keep_prob = %4.4f.',
                      self.hps.output_dropout_prob)
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=self.hps.output_dropout_prob)
    self.cell = cell
    =#
    sequence_lengths = Array{Int32}(undef, hps["batch_size"])
    input_data = Array{Float32}(undef, hps["batch_size"], hps["max_seq_len"]+1, 5)

    # The target/expected vectors of strokes
    output_x = input_data[:, 2:hps["max_seq_len"]+1, :]
    # vectors of strokes to be fed to decoder (same as above, but lagged behind
    # one step to include initial dummy value of (0, 0, 1, 0, 0))
    input_x = input_data[:, 1:hps["max_seq_len"], :]

    # either do vae-bit and get z, or do unconditional, decoder-only

    if hps["conditional"]
        #mean, presig = encoder()
        mean = 0.1
        #sigma = exp.(presig/2.0) # sigma > 0. div 2.0 -> sqrt.
        sigma = 1.01
        presig = 2*log(sigma)
        eps = randn(Float32, (hps["batch_size"], hps["z_size"]))
        batch_z = mean .+ sigma.*eps
        # KL cost
        kl_cost = -0.5*mean(1+presig-mean^2-exp(presig))
        kl_cost = max(kl_cost, hps["kl_tolerance"])
    else
        batch_z = zeros(Float32, (hps["batch_size"], hps["z_size"]))
        kl_cost = zeros(Float32, 1)
        actual_input_x = input_x
        #initial_state
    end


    #=
    # either do vae-bit and get z, or do unconditional, decoder-only
    if hps.conditional:  # vae mode:
      self.mean, self.presig = self.encoder(self.output_x,
                                            self.sequence_lengths)
      self.sigma = tf.exp(self.presig / 2.0)  # sigma > 0. div 2.0 -> sqrt.
      eps = tf.random_normal(
          (self.hps.batch_size, self.hps.z_size), 0.0, 1.0, dtype=tf.float32)
      self.batch_z = self.mean + tf.multiply(self.sigma, eps)
      # KL cost
      self.kl_cost = -0.5 * tf.reduce_mean(
          (1 + self.presig - tf.square(self.mean) - tf.exp(self.presig)))
      self.kl_cost = tf.maximum(self.kl_cost, self.hps.kl_tolerance)
      pre_tile_y = tf.reshape(self.batch_z,
                              [self.hps.batch_size, 1, self.hps.z_size])
      overlay_x = tf.tile(pre_tile_y, [1, self.hps.max_seq_len, 1])
      actual_input_x = tf.concat([self.input_x, overlay_x], 2)
      self.initial_state = tf.nn.tanh(
          rnn.super_linear(
              self.batch_z,
              cell.state_size,
              init_w='gaussian',
              weight_start=0.001,
              input_size=self.hps.z_size))
    else:  # unconditional, decoder-only generation
      self.batch_z = tf.zeros(
          (self.hps.batch_size, self.hps.z_size), dtype=tf.float32)
      self.kl_cost = tf.zeros([], dtype=tf.float32)
      actual_input_x = self.input_x
      self.initial_state = cell.zero_state(
          batch_size=hps.batch_size, dtype=tf.float32)
    =#

    num_mixture = hps["num_mixture"]

    # Number of outputs is 3 (one logit per pen state) plus 6 per mixture
    # component: mean_x, stdev_x, mean_y, stdev_y, correlation_xy, and the
    # mixture weight/probability (Pi_k)
    n_out = (3 + num_mixture * 6)

    #=
    with tf.variable_scope('RNN'):
      output_w = tf.get_variable('output_w', [self.hps.dec_rnn_size, n_out])
      output_b = tf.get_variable('output_b', [n_out])

    # decoder module of sketch-rnn is below
    output, last_state = tf.nn.dynamic_rnn(
        cell,
        actual_input_x,
        initial_state=self.initial_state,
        time_major=False,
        swap_memory=True,
        dtype=tf.float32,
        scope='RNN')

    output = tf.reshape(output, [-1, hps.dec_rnn_size])
    output = tf.nn.xw_plus_b(output, output_w, output_b)
    self.final_state = last_state
    =#

    function tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho)
      """Returns result of eq # 24 of http://arxiv.org/abs/1308.0850."""
      norm1 = x1 .- mu1
      norm2 = x2 .- mu2
      s1s2 = s1 .* s2
      # equation 25
      z = ((norm1 ./ s1).^2) + ((norm2 ./ s2).^2) -
           2 * ((rho .* (norm1 .* norm2)) ./ s1s2)
      neg_rho = 1 .- rho.^2
      result = exp.(-z ./ 2 * neg_rho)
      denom = 2 * pi * (s1s2 .* sqrt.(neg_rho))
      result = result ./ denom
      return result
    end

    function get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr,
                     z_pen_logits, x1_data, x2_data, pen_data)
      """Returns a loss fn based on eq #26 of http://arxiv.org/abs/1308.0850."""
      # This represents the L_R only (i.e. does not include the KL loss term).

      result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2,
                             z_corr)
      epsilon = 1e-6
      # result1 is the loss wrt pen offset (L_s in equation 9 of
      # https://arxiv.org/pdf/1704.03477.pdf)
      result1 = result0 .* z_pi
      result1 = sum.(result1) ## check this, it should do row sum
      result1 = -log.(result1 + epsilon)  # avoid log(0)

      fs = 1.0 - pen_data[:, 3]  # use training data for this
      fs = reshape(fs, (:, 1)) # check shape
      # Zero out loss terms beyond N_s, the last actual stroke
      result1 = result1 .* fs

      # result2: loss wrt pen state, (L_p in equation 9)
      result2 = nll(z_pen_logits, pen_data)
      result2 = reshape(result2, (:, 1))
      if !get_default_hparams()["is_training"]  # eval mode, mask eos columns
        result2 = result2 .* fs
      end

      result = result1 + result2
      return result
    end

    # below is where we need to do MDN (Mixture Density Network) splitting of
    # distribution params
    function get_mixture_coef(output)
      """Returns the tf slices containing mdn dist params."""
      # This uses eqns 18 -> 23 of http://arxiv.org/abs/1308.0850.
      z = output
      z_pen_logits = z[:, 1:3]  # pen states
      z_new = z[:, 4:end]
      n = 6
      z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr =  reshape(z_new, (n, div(size(z_new,2), n)))

      # process output z's into MDN parameters

      # softmax all the pi's and pen states:
      z_pi = softmax(z_pi)
      z_pen = softmax(z_pen_logits)

      # exponentiate the sigmas and also make corr between -1 and 1.
      z_sigma1 = exp.(z_sigma1)
      z_sigma2 = exp.(z_sigma2)
      z_corr = tanh.(z_corr)

      r = [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, z_pen_logits]
      return r
    end

    # out = get_mixture_coef(output)
    # println("out: ", out)
    # o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, o_pen_logits = out
    #
    #
    # target = reshape(output_x, (:, 5))
    # #split to 5 equal parts
    # x1_data, x2_data, eos_data, eoc_data, cont_data = reshape(target, (5, div(size(target, 2), 5)))
    # pen_data = hcat(eos_data, eoc_data, cont_data)
    #
    # lossfunc = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr,
    #                         o_pen_logits, x1_data, x2_data, pen_data)
    #
    # r_cost = mean(lossfunc)
    #
    # if hps["is_training"]
    #     lr = hps["learning_rate"]
    #     # optimizer = Adam
    #     kl_weight = hps["kl_weight_start"]
    #     cost = r_cost + kl_cost * kl_weight
    #
    #     #gradient clipping
    # end

    #=

    if self.hps.is_training:
      self.lr = tf.Variable(self.hps.learning_rate, trainable=False)
      optimizer = tf.train.AdamOptimizer(self.lr)

      self.kl_weight = tf.Variable(self.hps.kl_weight_start, trainable=False)
      self.cost = self.r_cost + self.kl_cost * self.kl_weight

      gvs = optimizer.compute_gradients(self.cost)
      g = self.hps.grad_clip
      capped_gvs = [(tf.clip_by_value(grad, -g, g), var) for grad, var in gvs]
      self.train_op = optimizer.apply_gradients(
          capped_gvs, global_step=self.global_step, name='train_step')
    =#
    return kl_cost
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
  if not model.hps.conditional:
    prev_state = sess.run(model.initial_state)
  else:
    prev_state = sess.run(model.initial_state, feed_dict={model.batch_z: z})
  =#

  strokes = zeros(Float32, seq_len, 5)
  mixture_params = []

  greedy = greedy_mode
  temp = temperature

  for i in 1:seq_len
    #=
      if not model.hps.conditional:
        feed = {
            model.input_x: prev_x,
            model.sequence_lengths: [1],
            model.initial_state: prev_state
        }
      else:
        feed = {
            model.input_x: prev_x,
            model.sequence_lengths: [1],
            model.initial_state: prev_state,
            model.batch_z: z
        }

      params = sess.run([
          model.pi, model.mu1, model.mu2, model.sigma1, model.sigma2, model.corr,
          model.pen, model.final_state], feed)
    =#

    o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state = params

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
