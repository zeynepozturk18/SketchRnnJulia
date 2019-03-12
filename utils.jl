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

struct DataLoader
    strokes
    batch_size
    max_seq_length
    scale_factor
    random_scale_factor
    augment_stroke_prob
    limit
end

function DataLoader(strokes; batchsize=100, dtype=Array{Array{Float32}},max_seq_length=250,
    scale_factor=1.0,
    random_scale_factor=0.0,
    augment_stroke_prob=0.0,
    limit=1000)
    DataLoader(strokes, batchsize, max_seq_length,
    scale_factor,
    random_scale_factor,
    augment_stroke_prob,
    limit)
end

function preprocess(d::DataLoader, strokes)
    """Remove entries from strokes having > max_seq_length points."""
    raw_data = []
    seq_len = []
    count_data = 0
    for i in 1:size(strokes,1)
      data = strokes[i]
      if size(data,1) <= (d.max_seq_length):
        count_data += 1
        # removes large gaps from the data
        data = min.(data, d.limit)
        data = max.(data, -d.limit)
        data = convert(Array{Float32}, data)
        data[:, 1:2] ./= d.scale_factor
        append!(raw_data, data)
        append!(seq_len, size(data,1))
      end
    end
    idx = sortperm(seq_len)
    d.strokes = []
    for i in 1:size(seq_len,1)
      append!(d.strokes,raw_data[idx[i]])
    end
    println("total images <= max_seq_len is $count_data")
    d.num_batches = divrem(count_data, d.batch_size)[1]
end
