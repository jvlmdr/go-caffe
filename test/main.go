package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-file/fileutil"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, os.Args[0], "model.json weights mean.npy mean-rgb layers image.(jpeg|png)")
		flag.PrintDefaults()
	}
}

func main() {
	var epsRel, epsAbs float64
	flag.Float64Var(&epsRel, "eps-rel", 1e-6, "Relative error threshold")
	flag.Float64Var(&epsAbs, "eps-abs", 1e-6, "Absolute error threshold")

	flag.Parse()
	if flag.NArg() != 7 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		extractScript = flag.Arg(0)
		modelFile     = flag.Arg(1)
		weightsFile   = flag.Arg(2)
		meanFile      = flag.Arg(3)
		meanStr       = flag.Arg(4)
		layersStr     = flag.Arg(5)
		imageFile     = flag.Arg(6)
	)

	model := new(caffe.NetParameter)
	if err := fileutil.LoadJSON(modelFile, model); err != nil {
		log.Fatalln("load model:", err)
	}
	mean, err := meanFromStr(meanStr)
	if err != nil {
		log.Fatalln("parse mean from args:", err)
	}
	layers := strings.Split(layersStr, ",")
	weights, err := loadWeights(weightsFile)
	if err != nil {
		log.Fatalln("load weights:", err)
	}
	// Load model again. Easier than deep copy.
	net := new(caffe.NetParameter)
	if err := fileutil.LoadJSON(modelFile, net); err != nil {
		log.Fatalln("load model:", err)
	}
	copyBlobs(net, weights)
	x, err := loadImage(imageFile)
	if err != nil {
		log.Fatalln("load image:", err)
	}

	var pass int
	for _, layer := range layers {
		log.Println("test layer:", layer)
		phi, err := caffe.FromProto(net, layer, mean)
		if err != nil {
			log.Fatalln("convert to feature transform:", err)
		}
		layerModel := caffe.SubsetForOutput(model, layer)
		log.Print("compute features using caffe")
		ys, err := caffe.Extract(extractScript, []image.Image{x}, layer, layerModel, weightsFile, meanFile)
		if err != nil {
			log.Fatalln("compute features using caffe:", err)
		}
		want := ys[0]
		log.Print("compute features in go")
		got, err := phi.Apply(x)
		if err != nil {
			log.Fatalln("compute features in go:", err)
		}
		log.Print("compare outputs")
		if !eq(want, got, epsRel, epsAbs) {
			continue
		}
		pass++
	}
	fmt.Printf("%d / %d tests pass\n", pass, len(layers))
}

func loadImage(fname string) (image.Image, error) {
	file, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	im, _, err := image.Decode(file)
	if err != nil {
		return nil, err
	}
	return im, nil
}

func meanFromStr(s string) ([]float64, error) {
	strs := strings.Split(s, ",")
	if len(strs) != 3 {
		return nil, fmt.Errorf("mean must have 3 elements: found %d", len(strs))
	}
	mean := make([]float64, len(strs))
	for i, str := range strs {
		x, err := strconv.ParseFloat(str, 64)
		if err != nil {
			return nil, err
		}
		mean[i] = x
	}
	return mean, nil
}

func loadWeights(fname string) (*caffe.NetParameter, error) {
	data, err := ioutil.ReadFile(fname)
	if err != nil {
		return nil, err
	}
	net := new(caffe.NetParameter)
	if err := proto.Unmarshal(data, net); err != nil {
		return nil, err
	}
	return net, nil
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

func eq(want, got *rimg64.Multi, epsRel, epsAbs float64) bool {
	if !got.Size().Eq(want.Size()) {
		log.Printf("size: want %v, got %v", want.Size(), got.Size())
		return false
	}
	if got.Channels != want.Channels {
		log.Printf("channels: want %d, got %d", want.Channels, got.Channels)
		return false
	}
	equal := true
	for i := 0; i < want.Width; i++ {
		for j := 0; j < want.Height; j++ {
			for k := 0; k < want.Channels; k++ {
				x, y := want.At(i, j, k), got.At(i, j, k)
				if math.Abs(x-y) <= epsRel*math.Abs(x) || math.Abs(x-y) <= epsAbs {
					continue
				}
				log.Printf("different at %d,%d,%d: want %g, got %g", i, j, k, x, y)
				equal = false
			}
		}
	}
	return equal
}
