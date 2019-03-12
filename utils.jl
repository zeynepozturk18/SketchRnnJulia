using Statistics

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

mutable struct DataLoader
    strokes
    batch_size
    max_seq_length
    scale_factor
    random_scale_factor
    augment_stroke_prob
    limit
    num_batches
    start_stroke_token
end

function DataLoader(strokes; batchsize=100, dtype=Array{Array{Float32}},
    max_seq_length=250,
    scale_factor=1.0,
    random_scale_factor=0.0,
    augment_stroke_prob=0.0,
    limit=1000, num_batches=0, start_stroke_token=[0, 0, 1, 0, 0])
    DataLoader(strokes, batchsize, max_seq_length,
    scale_factor,
    random_scale_factor,
    augment_stroke_prob,
    limit, num_batches, start_stroke_token)
end

function preprocess(d::DataLoader)
    """Remove entries from strokes having > max_seq_length points."""
    strokes = deepcopy(d.strokes)
    raw_data = []
    seq_len = []
    count_data = 0
    for i in 1:size(strokes,1)
      data = strokes[i]
      if size(data,1) <= (d.max_seq_length)
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

function calculate_normalizing_scale_factor(d::DataLoader)
  """Calculate the normalizing factor explained in appendix of sketch-rnn."""
  data = []
  for i in 1:size(d.strokes,1)
    if size(d.strokes[i],1) > d.max_seq_length
      continue
    end
    for j in 1:size(d.strokes[i],1)
      push!(data, d.strokes[i][j, 1])
      push!(data, d.strokes[i][j, 2])
    end
  end
  return std(data)
end
