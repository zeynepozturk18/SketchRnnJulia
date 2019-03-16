using Statistics: std
using Random

function get_max_len(strokes)
  """Return the maximum length of an array of strokes."""
  max_len = 0
  for stroke in strokes
    ml = size(stroke,1)
    if ml > max_len
      max_len = ml
    end
  end
  return max_len
end

function augment_strokes(strokes, prob=0.0)
  """Perform data augmentation by randomly dropping out strokes."""
  # drop each point within a line segments with a probability of prob
  # note that the logic in the loop prevents points at the ends to be dropped.
  result = []
  prev_stroke = [0, 0, 1]
  count = 0
  stroke = [0, 0, 1]  # Added to be safe.
  for i in 1:size(strokes,1)
      #println("iteration: $i")
      #println("strokes size: ", size(strokes))
    candidate = [strokes[i,1], strokes[i,2], strokes[i,3]]
    if candidate[3] == 1 || prev_stroke[3] == 1
      count = 0
    else
      count += 1
    end
    urnd = rand()  # uniform random variable
    if candidate[3] == 0 && prev_stroke[3] == 0 && count > 2 && urnd < prob
      stroke[1] += candidate[1]
      stroke[2] += candidate[2]
    else
      stroke = candidate
      prev_stroke = stroke
      push!(result, stroke)
    end
  end
  result = hcat(result...)'
  return result
end

mutable struct DataLoader
    strokes
    batch_size
    max_seq_len
    scale_factor
    random_scale_factor
    augment_stroke_prob
    limit
    num_batches
    start_stroke_token
end

function DataLoader(strokes; batch_size=100,
    max_seq_len=250,
    scale_factor=1.0,
    random_scale_factor=0.0,
    augment_stroke_prob=0.0,
    limit=1000, num_batches=0, start_stroke_token=[0, 0, 1, 0, 0])
    DataLoader(strokes, batch_size, max_seq_len,
    scale_factor,
    random_scale_factor,
    augment_stroke_prob,
    limit, num_batches, start_stroke_token)
end

function preprocess(d::DataLoader)
    """Remove entries from strokes having > max_seq_len points."""
    strokes = deepcopy(d.strokes)
    raw_data = []
    seq_len = []
    count_data = 0
    for i in 1:size(strokes,1)
      data = strokes[i]
      if size(data,1) <= (d.max_seq_len)
        count_data += 1
        # removes large gaps from the data
        data = min.(data, d.limit)
        data = max.(data, -d.limit)
        data = convert(Array{Float32}, data)
        data[:, 1:2] ./= d.scale_factor
        push!(raw_data, data)
        append!(seq_len, size(data,1))
      end
    end
    idx = sortperm(seq_len)
    d.strokes = []
    for i in 1:size(seq_len,1)
      push!(d.strokes,raw_data[idx[i]])
    end
    println("total images <= max_seq_len is $count_data")
    d.num_batches = divrem(count_data, d.batch_size)[1]
    println("nb of batches: ", d.num_batches)
    println("nb of strokes: ", size(d.strokes))
    return d
end

function random_sample(d::DataLoader)
  """Return a random sample, in stroke-3 format as used by draw_strokes."""
  sample = deepcopy(rand(d.strokes))
  return sample
end

function random_scale(d::DataLoader, data)
  """Augment data by stretching x and y axis randomly [1-e, 1+e]."""
  x_scale_factor = (
      rand() - 0.5) * 2 * d.random_scale_factor + 1.0
  y_scale_factor = (
      rand() - 0.5) * 2 * d.random_scale_factor + 1.0
  result = deepcopy(data)
  result[:, 1] .*= x_scale_factor
  result[:, 2] .*= y_scale_factor
  return result
end

function calculate_normalizing_scale_factor(d::DataLoader)
  """Calculate the normalizing factor explained in appendix of sketch-rnn."""
  data = []
  for i in 1:size(d.strokes,1)
    if size(d.strokes[i],1) > d.max_seq_len
      continue
    end
    for j in 1:size(d.strokes[i],1)
      push!(data, d.strokes[i][j, 1])
      push!(data, d.strokes[i][j, 2])
    end
  end
  return std(data)
end

function normalize(d::DataLoader, scale_factor=nothing)
  """Normalize entire dataset (delta_x, delta_y) by the scaling factor."""
  if !@isdefined scale_factor
    scale_factor = calculate_normalizing_scale_factor(d)
  d.scale_factor = scale_factor
  end
  for i in 1:size(d.strokes,1)
    d.strokes[i][:, 1:2] ./= d.scale_factor
  end
  return d
end

function _get_batch_from_indices(d::DataLoader, indices)
  """Given a list of indices, return the potentially augmented batch."""
  x_batch = []
  seq_len = []
  #println("indices size: ", size(indices))
  for idx in 1:size(indices,1)
    i = indices[idx]
    #println("i : ", i)
    data = random_scale(d, d.strokes[i])
    data_copy = deepcopy(data)
    if d.augment_stroke_prob > 0
      data_copy = augment_strokes(data_copy, d.augment_stroke_prob)
    end
    push!(x_batch,data_copy)
    length = size(data_copy,1)
    push!(seq_len,length)
  seq_len = convert(Array{Int64}, seq_len)
  end
  #println("size x_batch: " ,size(x_batch))
  # We return three things: stroke-3 format, stroke-5 format, list of seq_len.
  return x_batch, pad_batch(d, x_batch, d.max_seq_len), seq_len
end

function random_batch(d::DataLoader)
  """Return a randomised portion of the training data."""
  idx = randperm(size(d.strokes,1))[1:d.batch_size]
  return _get_batch_from_indices(d, idx)
end

function get_batch(d::DataLoader, idx)
  @assert idx >= 0, "idx must be non negative"
  @assert idx < d.num_batches, "idx must be less than the number of batches"
  start_idx = idx * d.batch_size
  indices = start_idx:(start_idx + d.batch_size-1)
  return _get_batch_from_indices(d, indices)
end

function pad_batch(d::DataLoader, batch, max_len)
  """Pad the batch to be stroke-5 bigger format as described in paper."""
  result = zeros(Float32, d.batch_size, max_len + 1, 5)
  @assert size(batch,1) == d.batch_size
  for i in 1:d.batch_size
    l = size(batch[i],1)
    #println("size x_batch l: " ,size(batch[i]))
    @assert l <= max_len
    result[i, 1:l, 1:2] = batch[i][:, 1:2]
    result[i, 1:l, 4] = batch[i][:, 3]
    result[i, 1:l, 3] = 1 .- result[i, 1:l, 4]
    result[i, l:end, 5] .= 1
    # put in the first token, as described in sketch-rnn methodology
    result[i, 2:end, :] = result[i, 1:end .!= end, :]
    result[i, 1, :] .= 0
    result[i, 1, 3] = d.start_stroke_token[3]  # setting S_0 from paper.
    result[i, 1, 4] = d.start_stroke_token[4]
    result[i, 1, 5] = d.start_stroke_token[5]
  end
  return result
end
