package main

import (
	"flag"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"io/ioutil"
	"log"
	"math/rand"
	"os"
	"strings"
	"time"

	"code.google.com/p/goprotobuf/proto"
	"github.com/jvlmdr/go-caffe/caffe"
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

	im, err := readImage(imFile)
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

	durs := make([][]float64, len(models))
	for i := range durs {
		durs[i] = make([]float64, numTrials)
	}
	for t := 0; t < numTrials; t++ {
		for i := range rand.Perm(len(models)) {
			start := time.Now()
			_, err := caffe.Extract(scriptFile, ims, outputs[i], models[i], weightsFile, meanFile)
			dur := time.Since(start)
			if err != nil {
				log.Fatalln(err)
			}
			durs[i][t] = dur.Seconds()
		}
	}

	for i := range outputs {
		fmt.Printf("%v\t%v\n", outputs[i], durs[i])
	}
}

func readImage(fname string) (image.Image, error) {
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
