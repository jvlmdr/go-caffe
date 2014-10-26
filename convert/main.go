package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"os"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
	"github.com/jvlmdr/go-file/fileutil"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s model.txt weights layer out.json\n", os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
	flag.Parse()
	if flag.NArg() != 4 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		modelFile   = flag.Arg(0)
		weightsFile = flag.Arg(1)
		layer       = flag.Arg(2)
		outFile     = flag.Arg(3)
	)

	data, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatal(err)
	}
	model := new(caffe.NetParameter)
	if err := proto.UnmarshalText(string(data), model); err != nil {
		log.Fatal(err)
	}

	data, err = ioutil.ReadFile(weightsFile)
	if err != nil {
		log.Fatal(err)
	}
	weights := new(caffe.NetParameter)
	if err := proto.Unmarshal(data, weights); err != nil {
		log.Fatal(err)
	}

	copyBlobs(model, weights)
	phi, err := caffe.FromProto(model, layer, []float64{0, 0, 0})
	if err != nil {
		log.Fatal(err)
	}
	if err := fileutil.SaveJSON(outFile, phi.Marshaler()); err != nil {
		log.Fatal(err)
	}
}

func copyBlobs(dst, src *caffe.NetParameter) {
	for _, dstLayer := range dst.Layers {
		name := dstLayer.GetName()
		srcLayer := findLayer(src, name)
		if srcLayer == nil {
			continue
		}
		dstLayer.Blobs = make([]*caffe.BlobProto, len(srcLayer.Blobs))
		copy(dstLayer.Blobs, srcLayer.Blobs)
	}
}

func findLayer(net *caffe.NetParameter, name string) *caffe.LayerParameter {
	for _, layer := range net.Layers {
		if layer.GetName() == name {
			return layer
		}
	}
	return nil
}
