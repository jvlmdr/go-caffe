package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	"image/png"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path"
	"strings"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
	"github.com/jvlmdr/go-cv/rimg64"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: %s extract.py image model.txt layers weights mean.npy\n", os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
	var numTrials int
	flag.IntVar(&numTrials, "trials", 16, "Number of trials for benchmark")
	flag.Parse()
	if flag.NArg() != 6 {
		flag.Usage()
		os.Exit(1)
	}
	var (
		scriptFile  = flag.Arg(0)
		imFile      = flag.Arg(1)
		modelFile   = flag.Arg(2)
		outputs     = strings.Split(flag.Arg(3), ",")
		weightsFile = flag.Arg(4)
		meanFile    = flag.Arg(5)
	)

	im, err := loadImage(imFile)
	if err != nil {
		log.Fatalln(err)
	}
	ims := []image.Image{im}

	// Load model.
	model := new(caffe.NetParameter)
	modelStr, err := ioutil.ReadFile(modelFile)
	if err != nil {
		log.Fatalln(err)
	}
	if err := proto.UnmarshalText(string(modelStr), model); err != nil {
		log.Fatalln(err)
	}

	// Extract subset of model for each layer.
	models := make([]*caffe.NetParameter, len(outputs))
	for i, output := range outputs {
		models[i] = caffe.SubsetForOutput(model, output)
		fmt.Printf("model for %s:\n", output)
		fmt.Println(models[i])
	}

	for i := range outputs {
		fs, err := caffe.Extract(scriptFile, ims, outputs[i], models[i], weightsFile, meanFile)
		if err != nil {
			log.Fatal(err)
		}
		f := fs[0]
		if err := visualize(f, outputs[i]); err != nil {
			log.Fatal(err)
		}
	}
}

func visualize(f *rimg64.Multi, name string) error {
	dir := name
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	// Save each channel as an image.
	for k := 0; k < f.Channels; k++ {
		im := rimg64.ToGray(normalize(f.Channel(k), 1e-3, true, false))
		fname := fmt.Sprintf("%d.png", k)
		if err := savePNG(path.Join(dir, fname), im); err != nil {
			return err
		}
	}
	// Create HTML file.
	file, err := os.Create(path.Join(dir, "index.html"))
	if err != nil {
		return err
	}
	defer file.Close()
	for k := 0; k < f.Channels; k++ {
		fmt.Fprintf(file, "<img src=\"%d.png\" />\n", k)
	}
	return nil
}

func square(x float64) float64 { return x * x }

func normMulti(f *rimg64.Multi, eps float64, pos bool, inv bool) *rimg64.Multi {
	min, max := -eps, +eps
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				min = math.Min(min, f.At(i, j, k))
				max = math.Max(max, f.At(i, j, k))
			}
		}
	}
	if pos {
		min = 0
	}
	if inv {
		min, max = max, min
	}
	g := rimg64.NewMulti(f.Width, f.Height, f.Channels)
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			for k := 0; k < f.Channels; k++ {
				g.Set(i, j, k, (f.At(i, j, k)-min)/(max-min))
			}
		}
	}
	return g
}

func normalize(f *rimg64.Image, eps float64, pos bool, inv bool) *rimg64.Image {
	min, max := -eps, +eps
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			min = math.Min(min, f.At(i, j))
			max = math.Max(max, f.At(i, j))
		}
	}
	if pos {
		min = 0
	}
	if inv {
		min, max = max, min
	}
	g := rimg64.New(f.Width, f.Height)
	for i := 0; i < f.Width; i++ {
		for j := 0; j < f.Height; j++ {
			g.Set(i, j, (f.At(i, j)-min)/(max-min))
		}
	}
	return g
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

func savePNG(fname string, im image.Image) error {
	file, err := os.Create(fname)
	if err != nil {
		return err
	}
	defer file.Close()
	return png.Encode(file, im)
}
