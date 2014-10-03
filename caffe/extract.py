import argparse
import numpy as np
import caffe
import csv
from google.protobuf import text_format
from caffe.proto import caffe_pb2
import tempfile
import os

def load_mean(fname):
  return np.mean(np.load(fname), (1,2))

def read_csv(fname):
  with open(fname, "rb") as f:
    reader = csv.reader(f)
    return [x for x in reader]

def save_image(fname, arr):
  with open(fname, "w") as f:
    writer = csv.writer(f)
    c, m, n = arr.shape
    for k in range(c):
      for i in range(m):
        for j in range(n):
          writer.writerow([j, i, k, repr(arr[k,i,j])])

def ceildiv(a, b):
  return (a + b - 1) / b

def layer_size(net, name, size):
  # Traverse back to "data".
  layers = {l.name: l for l in net.layers}
  def helper(name):
    if name == "data":
      return size
    if name not in layers:
      raise RuntimeError("layer not found: " + name)
    layer = layers[name]

    field, stride = 1, 1
    if layer.type == caffe_pb2.LayerParameter.CONVOLUTION:
      field = layer.convolution_param.kernel_size
      stride = layer.convolution_param.stride
    elif layer.type == caffe_pb2.LayerParameter.POOLING:
      field = layer.pooling_param.kernel_size
      stride = layer.pooling_param.stride
    elif layer.type == caffe_pb2.LayerParameter.LRN:
      pass
    else:
      enum = caffe_pb2.LayerParameter.DESCRIPTOR.enum_types_by_name["LayerType"]
      value = enum.values_by_number[layer.type].name
      raise RuntimeError("unknown layer type: " + value)

    bottoms = layer.bottom
    if len(bottoms) != 1:
      raise RuntimeError("number of input layers not one: " + len(bottoms))
    prev = helper(bottoms[0])
    out = tuple([ceildiv(n-field+1, stride) for n in prev])
    print("{}: {} -> {}".format(name, prev, out))
    return out
  return helper(name)

def preprocess(net, input_name, im, mean):
    """
    Format input for Caffe:
    - convert to single
    - reorder channels (for instance color to BGR)
    - scale raw input (e.g. from [0, 1] to [0, 255] for ImageNet models)
    - transpose dimensions to K x H x W
    - subtract mean
    - scale feature

    Take
    input_name: name of input blob to preprocess for
    im: (H' x W' x K) ndarray

    Give
    caffe_inputs: (K x H x W) ndarray
    """
    out = im.astype(np.float32, copy=False)
    input_scale = net.input_scale.get(input_name)
    raw_scale = net.raw_scale.get(input_name)
    channel_order = net.channel_swap.get(input_name)
    out -= np.tile(mean, (out.shape[0], out.shape[1], 1))
    # in_size = net.blobs[input_name].data.shape[2:]
    # if out.shape[:2] != in_size:
    #     out = caffe.io.resize_image(out, in_size)
    if channel_order is not None:
        out = out[:, :, channel_order]
    out = out.transpose((2, 0, 1))
    if raw_scale is not None:
        out *= raw_scale
    if input_scale is not None:
        out *= input_scale
    return out

def copy_weights(dst, src):
  for l in dst.params:
    if len(dst.params[l]) != len(src.params[l]):
      raise RuntimeError("different number of params in layer " + l)
    for i in range(len(dst.params[l])):
      dst.params[l][i].data[:] = src.params[l][i].data

def save_model_temp(model):
  tmpfile, tmpname = tempfile.mkstemp()
  os.write(tmpfile, text_format.MessageToString(model))
  os.close(tmpfile)
  return tmpname

def load_model(fname):
  model = caffe_pb2.NetParameter()
  with open(fname) as f:
    text_format.Merge(f.read(), model)
  return model

def new_net(model):
  tmpname = save_model_temp(model)
  net = caffe.Net(tmpname)
  os.remove(tmpname)
  return net

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("model", metavar="model.prototxt")
  parser.add_argument("pretrained")
  parser.add_argument("mean", metavar="mean.npy")
  parser.add_argument("layer")
  parser.add_argument("files", metavar="files.csv")
  args = parser.parse_args()

  # Load input and output files from CSV.
  files = read_csv(args.files)
  # Load mean.
  mean = load_mean(args.mean)

  model = load_model(args.model)
  pretrained = caffe.Classifier(args.model, args.pretrained,
      channel_swap=(2,1,0), raw_scale=255.0)

  for in_file, out_file in files:
    # Get image size.
    im = caffe.io.load_image(in_file)
    imsz = (im.shape[0], im.shape[1])
    # Modify model to have image size.
    model.input_dim[0] = 1
    model.input_dim[2:4] = imsz
    # Instantiate network.
    net = new_net(model)
    copy_weights(net, pretrained)
    net.set_phase_test()
    net.set_channel_swap("data", (2,1,0))
    net.set_raw_scale("data", 255.0)
    # Evaluate network.
    data = np.asarray([preprocess(net, "data", im, mean)])
    net.forward(data=data)
    out = net.blobs[args.layer].data
    # Take valid sub-image.
    ftsz = layer_size(model, args.layer, imsz)
    print("crop {} from {}".format(ftsz, out.shape[2:4]))
    out = out[0, :, :ftsz[0], :ftsz[1]]
    save_image(out_file, out)

if __name__ == "__main__":
  main()
