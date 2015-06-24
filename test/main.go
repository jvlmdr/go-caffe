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
	"time"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
	"github.com/jvlmdr/go-cv/rimg64"
	"github.com/jvlmdr/go-file/fileutil"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, os.Args[0], "extract.py arch.json weights mean.npy mean-rgb layer image.(jpeg|png)")
		flag.PrintDefaults()
	}
}

func main() {
	var (
		epsRel float64
		epsAbs float64
		trials int
	)
	flag.Float64Var(&epsRel, "eps-rel", 1e-6, "Relative error threshold")
	flag.Float64Var(&epsAbs, "eps-abs", 1e-6, "Absolute error threshold")
	flag.IntVar(&trials, "trials", 16, "Number of trials for benchmark")

	flag.Parse()
	if flag.NArg() != 7 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		script      = flag.Arg(0)
		archFile    = flag.Arg(1)
		weightsFile = flag.Arg(2)
		meanFile    = flag.Arg(3)
		meanStr     = flag.Arg(4)
		layer       = flag.Arg(5)
		imageFile   = flag.Arg(6)
	)

	arch := new(caffe.NetParameter)
	if err := fileutil.LoadJSON(archFile, arch); err != nil {
		log.Fatalln("load architecture:", err)
	}
	mean, err := meanFromStr(meanStr)
	if err != nil {
		log.Fatalln("parse mean from args:", err)
	}
	weights, err := loadWeights(weightsFile)
	if err != nil {
		log.Fatalln("load weights:", err)
	}
	// Load model again. Easier than deep copy.
	net := new(caffe.NetParameter)
	if err := fileutil.LoadJSON(archFile, net); err != nil {
		log.Fatalln("load architecture:", err)
	}
	copyBlobs(net, weights)
	im, err := loadImage(imageFile)
	if err != nil {
		log.Fatalln("load image:", err)
	}
	err = test(im, net, layer, mean, script, arch, weightsFile, meanFile, epsRel, epsAbs)
	if err != nil {
		log.Fatal(err)
	}
	err = bench(im, net, layer, mean, script, arch, weightsFile, meanFile, trials)
	if err != nil {
		log.Fatal(err)
	}
}

// net is a populated network, which will be converted to a native feature transform.
// arch is the empty network whose architecture will be used to load weightsFile.
func test(im image.Image, net *caffe.NetParameter, layer string, mean []float64, script string, arch *caffe.NetParameter, weightsFile, meanFile string, epsRel, epsAbs float64) error {
	log.Println("test layer:", layer)
	phi, err := caffe.FromProto(net, layer, mean)
	if err != nil {
		log.Fatalln("convert to feature transform:", err)
	}
	// Take the architecture subset necessary to compute this layer.
	subset := caffe.SubsetForOutput(arch, layer)
	log.Print("compute features using caffe")
	ys, err := caffe.Extract(script, []image.Image{im}, layer, subset, weightsFile, meanFile)
	if err != nil {
		return fmt.Errorf("compute features using caffe: %v", err)
	}
	want := ys[0]
	log.Print("compute features in go")
	got, err := phi.Apply(im)
	if err != nil {
		log.Fatalln("compute features in go:", err)
	}
	log.Print("compare outputs")
	if !eq(want, got, epsRel, epsAbs) {
		fmt.Println("FAIL")
	} else {
		fmt.Println("PASS")
	}
	return nil
}

// net is a populated network, which will be converted to a native feature transform.
// arch is the empty network whose architecture will be used to load weightsFile.
func bench(im image.Image, net *caffe.NetParameter, layer string, mean []float64, script string, arch *caffe.NetParameter, weightsFile, meanFile string, trials int) error {
	log.Println("test layer:", layer)
	phi, err := caffe.FromProto(net, layer, mean)
	if err != nil {
		log.Fatalln("convert to feature transform:", err)
	}
	var durPython, durNative float64
	for i := 0; i < trials; i++ {
		// Take the architecture subset necessary to compute this layer.
		subset := caffe.SubsetForOutput(arch, layer)
		log.Print("compute features using caffe")
		start := time.Now()
		_, err = caffe.Extract(script, []image.Image{im}, layer, subset, weightsFile, meanFile)
		durPython += time.Since(start).Seconds()
		if err != nil {
			return fmt.Errorf("compute features using caffe: %v", err)
		}
		log.Print("compute features in go")
		start = time.Now()
		_, err = phi.Apply(im)
		if err != nil {
			log.Fatalln("compute features in go:", err)
		}
		durNative += time.Since(start).Seconds()
	}
	fmt.Printf("Python: %.3g sec\n", durPython/float64(trials))
	fmt.Printf("native: %.3g sec\n", durNative/float64(trials))
	return nil
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
